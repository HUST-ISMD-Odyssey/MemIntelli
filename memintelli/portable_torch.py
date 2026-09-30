"""Array current, ADC and signed reconstruction using only PyTorch."""
import math
import warnings
import torch


def adc_round(value):
    """Stabilize FP32 round-to-even at half-code ties, without integer decoding."""
    half = (value * 2).round() * .5
    tolerance = (value.abs().clamp_min(1) * (4 * torch.finfo(torch.float32).eps)).clamp_max(1e-4)
    value = torch.where((value-half).abs() <= tolerance, half, value)
    return value.round()


def adc_transfer(signal, maximum, bits, clip=True, stabilize_ties=True):
    """Return reconstructed integer partial sums, not ADC code numbers."""
    codes = 2**bits - 1
    if clip:
        exponent = max(1, math.ceil(math.log2(maximum)))
        step = 2**(exponent - bits)
        limit = min(maximum, 2**exponent - 1)
        highest = min(codes, math.floor(limit / step))
        value = signal / step
        code = adc_round(value) if stabilize_ties else value.round()
        return code.clamp(0, highest) * step
    value = signal / maximum * codes
    code = adc_round(value) if stabilize_ties else value.round()
    return code.clamp(0, codes) / codes * maximum


def _slice_factors(slices):
    maxima, significance = [], []
    shift = 0
    for width in reversed(slices):
        maxima.append(2**width - 1)
        significance.append(2**shift)
        shift += width
    significance[-1] *= -1
    return maxima, significance


def accumulate(x, conductance, mat_max, mat, engine, reference=None, valid_columns=None):
    if (engine.torch_fuse_adc and x.sliced_data.is_cuda and engine.execution_mode == "speed" and engine.adc_clip
            and engine.dac_bits == 1 and reference is None
            and max(engine.input_slice + engine.weight_slice) == 1):
        return _binary_accumulate(x, conductance, mat_max, mat, engine, valid_columns)
    return _accumulate_reference(x, conductance, mat_max, mat, engine, reference, valid_columns)


def _binary_accumulate(x, conductance, mat_max, mat, engine, valid_columns):
    """Join independent weight-slice columns, keeping every per-array ADC."""
    if (engine.adc_bits <= 8 and x.sliced_data.dtype == torch.uint8
            and max(*x.sliced_data.shape, *x.sliced_data.stride()) < 2**31
            and (2**engine.adc_bits-1)*(2**x.total_bits-1)*(2**mat.total_bits-1) <= 2**24):
        result = _binary_grouped_accumulate(x, conductance, mat_max, mat, engine, valid_columns)
        if result is not None:
            return result
    fused = None
    if len(engine.weight_slice) <= 7:
        fused = _get_fused_adc(engine)
    if (fused is None and engine.adc_bits <= 8
            and (2**engine.adc_bits-1)*(2**x.total_bits-1)*(2**mat.total_bits-1) <= 2**24):
        return _binary_lookup_accumulate(x, conductance, mat_max, mat, engine, valid_columns)
    xs = x.sliced_data
    n, m, ni, j, k = xs.shape
    _, p, ns, _, l = conductance.shape
    if j != 1:
        raise ValueError("Input array height must be one")
    columns = p*l if valid_columns is None else valid_columns
    g = conductance.to(torch.bfloat16).permute(0, 3, 2, 1, 4).reshape(m, k, ns, p*l)
    g = g[..., :columns].reshape(m, k, ns*columns).contiguous()
    partial = torch.zeros((m, n, columns), device=xs.device, dtype=torch.float32)
    adc_ref = (engine.HGS-engine.LGS)*engine.vread*k
    exponent = max(1, math.ceil(math.log2(k)))
    step = 2**(exponent-engine.adc_bits)
    highest = min(2**engine.adc_bits-1, math.floor(min(k, 2**exponent-1)/step))
    _, xsign = _slice_factors(engine.input_slice)
    _, wsign = _slice_factors(engine.weight_slice)
    # Bound the expanded slice-current workspace independently of array geometry.
    row_chunk = max(1, min(n, 16_777_216//(m*ns*columns)))
    old_reduce = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        for i in range(ni):
            for begin in range(0, n, row_chunk):
                end = min(begin+row_chunk, n)
                raw = xs[begin:end, :, i, 0].permute(1, 0, 2).float()
                v = (raw*engine.vread).to(torch.bfloat16)
                current = torch.bmm(v, g).reshape(m, end-begin, ns, columns)
                target = partial[:, begin:end]
                if fused is not None:
                    try:
                        target.copy_(fused(*current.unbind(2), target, significance=float(xsign[i])))
                        engine._torch_fusion_status = "active"
                        continue
                    except RuntimeError as exc:
                        if isinstance(exc, torch.cuda.OutOfMemoryError) or not any(
                            word in str(exc).lower() for word in ("nvrtc", "jiterator", "compile")
                        ):
                            raise
                        _disable_fusion(engine, exc)
                        fused = None
                current = current.float()
                current.div_(adc_ref).mul_(k).div_(step).round_().clamp_(0, highest).mul_(step)
                for s in range(ns):
                    target.add_(current[:, :, s], alpha=xsign[i]*wsign[s])
        xm = x.max_data.reshape(n, m)
        mm = mat_max.expand(m, p, 1, l).reshape(m, p*l)
        partial *= xm.T[:, :, None]*mm[:, None, :columns]
        partial /= 2**(x.total_bits-1)-1
        partial /= 2**(mat.total_bits-1)-1
        return partial.sum(0)
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_reduce


def _disable_fusion(engine, error):
    engine._torch_fusion_status = "unavailable"
    detail = "\n".join(str(error).strip().splitlines()[-3:])[-1000:]
    engine._torch_fusion_error = detail
    warnings.warn(f"Torch CUDA ADC fusion unavailable; using portable Torch: {detail}",
                  RuntimeWarning, stacklevel=3)


def _get_fused_adc(engine):
    if getattr(engine, "_torch_fusion_status", None) == "unavailable":
        return None
    if hasattr(engine, "_torch_adc_function"):
        return engine._torch_adc_function
    try:
        from torch.cuda.jiterator import _create_jit_fn
    except ImportError as exc:
        _disable_fusion(engine, exc)
        return None
    _, significance = _slice_factors(engine.weight_slice)
    arguments = ", ".join(f"T c{i}" for i in range(len(significance)))
    body = ["float result = float(base);"]
    for i, scale in enumerate(significance):
        body.extend([
            f"float q{i} = nearbyintf(__fmul_rn(__fmul_rn(__fmul_rn(float(c{i}), "
            "float(inv_ref)), float(rows)), float(inv_step)));",
            f"q{i} = q{i} < 0.0f ? 0.0f : (q{i} > float(highest) ? float(highest) : q{i});",
            f"float term{i} = __fmul_rn(__fmul_rn(q{i}, float(step)), "
            f"__fmul_rn(float(significance), {float(scale)}f));",
            f"result = __fadd_rn(result, term{i});",
        ])
    body.append("return T(result);")
    code = (f"template <typename T> T memintelli_adc({arguments}, T base, T inv_ref, "
            "T rows, T inv_step, T highest, T step, T significance) {"
            + "\n".join(body) + "}")
    rows = engine.weight_paral_size[0]
    exponent = max(1, math.ceil(math.log2(rows)))
    step = 2**(exponent-engine.adc_bits)
    adc_ref = (engine.HGS-engine.LGS)*engine.vread*rows
    inverse = torch.tensor(adc_ref, dtype=torch.float32).reciprocal().item()
    engine._torch_adc_function = _create_jit_fn(
        code, inv_ref=inverse, rows=float(rows), inv_step=1.0/step,
        highest=float(min(2**engine.adc_bits-1, math.floor(min(rows, 2**exponent-1)/step))),
        step=float(step), significance=1.0,
    )
    return engine._torch_adc_function


def _binary_grouped_accumulate(x, conductance, mat_max, mat, engine, valid_columns):
    """Batch input slices only when reassociation is exact in the ADC code domain."""
    from .portable_cuda import get_kernel
    fused = get_kernel(engine, "adc_reduce")
    voltage = get_kernel(engine, "voltage") if fused is not None else None
    if voltage is None:
        return None
    xs = x.sliced_data
    n, m, ni, j, k = xs.shape
    _, p, ns, _, l = conductance.shape
    if j != 1:
        raise ValueError("Input array height must be one")
    columns = p*l if valid_columns is None else valid_columns
    g = conductance.to(torch.bfloat16).permute(0, 3, 2, 1, 4).reshape(m, k, ns, p*l)
    g = g[..., :columns].reshape(m, k, ns*columns).contiguous()
    partial = torch.empty((m, n, columns), device=xs.device, dtype=torch.float32)
    xm = x.max_data.reshape(n, m).float().contiguous()
    mm = mat_max.expand(m, p, 1, l).reshape(m, p*l)[:, :columns].float().contiguous()
    # Keep the materialized BF16 slice currents near 32 MiB to limit memory traffic.
    row_chunk = max(1, min(n, 16_777_216//(m*ni*ns*columns)))
    old_reduce = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        for begin in range(0, n, row_chunk):
            end = min(begin+row_chunk, n)
            v = torch.empty((m, ni*(end-begin), k), device=xs.device, dtype=torch.bfloat16)
            with torch.cuda.device(xs.device):
                voltage(grid=((v.numel()+255)//256, 1, 1), block=(256, 1, 1),
                        args=[xs, v, m, end-begin, begin, xs.stride(0), xs.stride(1),
                              xs.stride(2), xs.stride(4)])
            currents = torch.bmm(v, g).reshape(m, ni, end-begin, ns, columns)
            with torch.cuda.device(xs.device):
                fused(grid=((m*(end-begin)*columns+255)//256, 1, 1), block=(256, 1, 1),
                      args=[currents, xm, mm, partial, m, end-begin, n, columns, begin])
        engine._torch_fusion_status = "active"
        engine._torch_cuda_fusion_status = "active"
        return partial.sum(0)
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_reduce


def _binary_lookup_accumulate(x, conductance, mat_max, mat, engine, valid_columns):
    """Exact BF16-current ADC lookup; bounded integer sums are exact in FP32."""
    xs = x.sliced_data
    n, m, ni, j, k = xs.shape
    if j != 1:
        raise ValueError("Input array height must be one")
    _, p, ns, _, l = conductance.shape
    columns = p*l if valid_columns is None else valid_columns
    step = 2**(max(1, math.ceil(math.log2(k)))-engine.adc_bits)
    if not hasattr(engine, "_binary_adc_table"):
        patterns = torch.arange(65536, device=xs.device, dtype=torch.int32).to(torch.int16)
        currents = patterns.view(torch.bfloat16).float()
        adc_ref = (engine.HGS-engine.LGS)*engine.vread*k
        engine._binary_adc_table = (adc_transfer(
            currents/adc_ref*k, k, engine.adc_bits, stabilize_ties=False)/step).to(torch.bfloat16)
    g = conductance.to(torch.bfloat16).permute(0, 3, 2, 1, 4).reshape(m, k, ns, p*l)
    g = g[..., :columns].reshape(m, k, ns*columns).contiguous()
    significance = engine._slice_scales.to(torch.bfloat16)[None, :, None, :, None]
    partial = torch.empty((m, n, columns), device=xs.device, dtype=torch.float32)
    if engine._torch_fusion_status != "unavailable":
        engine._torch_fusion_status = "lookup"
    row_chunk = max(1, min(n, 16_777_216//(m*ni*ns*columns)))
    old_reduce = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        for begin in range(0, n, row_chunk):
            end = min(begin+row_chunk, n)
            raw = xs[begin:end, :, :, 0].permute(1, 2, 0, 3).reshape(m, ni*(end-begin), k)
            v = (raw.float()*engine.vread).to(torch.bfloat16)
            currents = torch.bmm(v, g)
            codes = engine._binary_adc_table[currents.view(torch.int16).to(torch.int32)]
            codes = codes.reshape(m, ni, end-begin, ns, columns)
            codes.mul_(significance)
            partial[:, begin:end] = codes.sum((1, 3), dtype=torch.float32)*step
        xm = x.max_data.reshape(n, m)
        mm = mat_max.expand(m, p, 1, l).reshape(m, p*l)
        partial *= xm.T[:, :, None]*mm[:, None, :columns]
        partial /= 2**(x.total_bits-1)-1
        partial /= 2**(mat.total_bits-1)-1
        return partial.sum(0)
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_reduce


def _accumulate_reference(x, conductance, mat_max, mat, engine, reference=None, valid_columns=None):
    xs = x.sliced_data
    n, m, ni, j, k = xs.shape
    _, p, ns, _, l = conductance.shape
    if j != 1:
        raise ValueError("Input array height must be one")
    columns = p*l if valid_columns is None else valid_columns
    partial = torch.zeros((m, n, columns), device=xs.device, dtype=torch.float32)
    adc_ref = (engine.HGS - engine.LGS) * engine.vread * k
    rdac, radc = engine.rdac - 1, engine.radc - 1
    xmax, xsign = _slice_factors(engine.input_slice)
    wmax, wsign = _slice_factors(engine.weight_slice)
    speed = engine.execution_mode == "speed"
    operand_g = conductance.to(torch.bfloat16) if speed else conductance
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    old_bf16_reduce = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        for i in range(ni):
            raw = xs[:, :, i, 0].permute(1, 0, 2).float()
            v = torch.round(raw / xmax[i] * rdac) * (engine.vread / rdac)
            v = engine._round_internal(v)
            if speed:
                v = v.to(torch.bfloat16)
            for s in range(ns):
                g = operand_g[:, :, s].permute(0, 2, 1, 3).reshape(m, k, p*l)[:, :, :columns]
                current = torch.bmm(v, g).float()
                if engine.adc_clip:
                    maximum = k * xmax[i] * wmax[s]
                    signal = current / adc_ref * maximum
                    if reference is not None:
                        gain = reference[:, :, s, 0].repeat_interleave(l, dim=1)[:, None, :columns]
                        base = reference[:, :, s, 1].repeat_interleave(l, dim=1)[:, None, :columns]
                        signal = (signal - base * raw.sum(-1, keepdim=True)) / gain
                    q = adc_transfer(signal, maximum, engine.adc_bits, stabilize_ties=not speed)
                    scale = xsign[i] * wsign[s]
                else:
                    value = current / adc_ref * radc
                    q = (value.round() if speed else adc_round(value)).clamp(0, radc) / radc
                    scale = xmax[i] * xsign[i] * wmax[s] * wsign[s] * k
                partial += q * scale
        xm = x.max_data.reshape(n, m)
        mm = mat_max.expand(m, p, 1, l).reshape(m, p*l)
        bm = xm.T[:, :, None] * mm[:, None, :columns]
        partial = partial * bm
        partial = partial / (2**(x.total_bits-1)-1)
        partial = partial / (2**(mat.total_bits-1)-1)
        return partial.sum(0)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = old_bf16_reduce
