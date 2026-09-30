"""Array current, ADC and signed reconstruction using only PyTorch."""
import math
import torch


def adc_transfer(signal, maximum, bits, clip=True):
    """Return reconstructed integer partial sums, not ADC code numbers."""
    codes = 2**bits - 1
    if clip:
        exponent = max(1, math.ceil(math.log2(maximum)))
        step = 2**(exponent - bits)
        limit = min(maximum, 2**exponent - 1)
        highest = min(codes, math.floor(limit / step))
        return (signal / step).round().clamp(0, highest) * step
    return (signal / maximum * codes).round().clamp(0, codes) / codes * maximum


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
                    q = adc_transfer(signal, maximum, engine.adc_bits)
                    scale = xsign[i] * wsign[s]
                else:
                    q = (current / adc_ref * radc).round().clamp(0, radc) / radc
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
