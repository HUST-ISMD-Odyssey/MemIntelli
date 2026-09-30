"""Optional Triton backend. Imported only when selected or attempted."""
import torch
import triton

from .pimpy.triton_bf16_kernel import _bf16_partial_kernel


def accumulate(x, conductance, mat_max, mat, engine, reference=None, valid_columns=None):
    xs = x.sliced_data.contiguous()
    g = conductance.contiguous()
    n, m, i, j, k = xs.shape
    _, p, s, _, l = g.shape
    if p == 1 and valid_columns is not None and valid_columns < l:
        g = g[..., :valid_columns].contiguous()
        l = valid_columns
    xm, mm = x.max_data.contiguous(), mat_max.contiguous()
    scales = (x.sliced_max_weights[:, None] * x.sliced_weights[:, None]
              * mat.sliced_max_weights[None, :] * mat.sliced_weights[None, :] * k).contiguous()
    if engine.adc_clip:
        scales = (x.sliced_weights[:, None] * mat.sliced_weights[None, :]).contiguous()
    partial = torch.empty((m, n*j, p*l), device=xs.device, dtype=torch.float32)
    nr, nl = triton.cdiv(n*j, 32), triton.cdiv(l, 32)
    with torch.cuda.device(xs.device):
        _bf16_partial_kernel[(nr*m*p*nl,)](
            xs, g, x.sliced_max_weights, scales, xm, mm, partial,
            *xs.stride(), *g.stride(), *scales.stride(),
            xm.stride(0), xm.stride(1), mm.stride(0), mm.stride(1), *partial.stride(),
            n, m, i, p, j, k, l, s,
            (engine.HGS-engine.LGS)*engine.vread*k, float(engine.rdac-1), float(engine.radc-1),
            engine.vread, float(2**(x.total_bits-1)-1), float(2**(mat.total_bits-1)-1),
            2 if engine.execution_mode == "speed" else 0, "ieee",
            x.is_uniform_1bit_slices, 32, 32, 32, nr, nl, m, 1,
            COUNT_ADC=engine.adc_clip, ADC_MAX_CODE=engine.radc-1,
            REFERENCE=reference, APPLY_REFERENCE=reference is not None,
            BF16_CURRENT=engine.execution_mode == "speed", CLAMP_NORMALIZED=True,
            INTEGER_ADC=True, WEIGHT_SLICE_MAX=mat.sliced_max_weights,
            COLUMN_SCALES=mm.shape[-1] != 1, MM_COLUMN_STRIDE=mm.stride(-1),
            num_warps=4, enable_fp_fusion=False,
        )
    return partial.sum(0)
