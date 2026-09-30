"""Optional Triton backend. Imported only when selected or attempted."""
import torch
import triton
import triton.language as tl

from .pimpy.triton_bf16_kernel import _bf16_partial_kernel
from .pimpy.triton_fast_accumulate import _round_even


@triton.jit
def _slice_input_kernel(X, BITS, MAXIMA, ROWS: tl.constexpr, FEATURES: tl.constexpr,
                        STRIDE0: tl.constexpr, STRIDE1: tl.constexpr,
                        ARRAY: tl.constexpr, GROUP: tl.constexpr, BLOCKS: tl.constexpr,
                        WIDTHS: tl.constexpr, QMAX: tl.constexpr, BR: tl.constexpr,
                        BK: tl.constexpr):
    rows = tl.program_id(0)*BR + tl.arange(0, BR)
    offsets = tl.arange(0, BK)
    cols = tl.program_id(1)*GROUP + offsets
    values = tl.load(X + rows[:, None]*STRIDE0 + cols[None, :]*STRIDE1,
                     (rows[:, None] < ROWS) & (offsets[None, :] < GROUP)
                     & (cols[None, :] < FEATURES), other=0.0)
    maximum = tl.max(tl.abs(values), 1)
    scaled = tl.div_rn(values, tl.where(maximum > 0, maximum, 1.0)[:, None])*QMAX
    quantized = _round_even(scaled).to(tl.int32)
    block, col = cols//ARRAY, cols % ARRAY
    mask = ((rows[:, None] < ROWS) & (offsets[None, :] < GROUP)
            & (block[None, :] < BLOCKS))
    shift = 0
    for i in tl.static_range(len(WIDTHS)):
        bits = (quantized >> shift) & ((1 << WIDTHS[i])-1)
        address = ((rows[:, None]*BLOCKS + block[None, :])*len(WIDTHS)+i)*ARRAY + col[None, :]
        tl.store(BITS + address, bits, mask)
        shift += WIDTHS[i]
    tl.store(MAXIMA + rows[:, None]*BLOCKS + block[None, :],
             tl.broadcast_to(maximum[:, None], (BR, BK)), mask & (col[None, :] == 0))


def slice_input(inputs, engine):
    group = engine.input_quant_gran[1]
    if group > 4096:
        return None
    n, features = inputs.shape
    size = engine.weight_paral_size[0]
    blocks = triton.cdiv(features, size)
    bits = torch.empty((n, blocks, len(engine.input_slice), 1, size),
                       device=inputs.device, dtype=torch.uint8)
    maxima = torch.empty((n, blocks, 1, 1), device=inputs.device, dtype=torch.float32)
    with torch.cuda.device(inputs.device):
        _slice_input_kernel[(triton.cdiv(n, 4), triton.cdiv(features, group))](
            inputs, bits, maxima, n, features, *inputs.stride(), size, group, blocks,
            tuple(reversed(engine.input_slice)), 2**(engine.activation_bits-1)-1,
            4, triton.next_power_of_2(group), num_warps=4, enable_fp_fusion=False,
        )
    return bits, maxima


def accumulate(x, conductance, mat_max, mat, engine, reference=None, valid_columns=None):
    xs = x.sliced_data.contiguous()
    g = conductance.contiguous()
    n, m, i, j, k = xs.shape
    _, p, s, _, l = g.shape
    if p == 1 and valid_columns is not None and valid_columns < l:
        g = g[..., :valid_columns].contiguous()
        l = valid_columns
    xm, mm = x.max_data.contiguous(), mat_max.contiguous()
    scales = engine._slice_scales
    partial = torch.empty((m, n*j, p*l), device=xs.device, dtype=torch.float32)
    tuned = engine.execution_mode == "speed" and k == 64 and i*s <= 64
    block_r, block_k = (64, 64) if tuned else (32, 32)
    nr, nl = triton.cdiv(n*j, block_r), triton.cdiv(l, 32)
    with torch.cuda.device(xs.device):
        _bf16_partial_kernel[(nr*m*p*nl,)](
            xs, g, x.sliced_max_weights, scales, xm, mm, partial,
            *xs.stride(), *g.stride(), *scales.stride(),
            xm.stride(0), xm.stride(1), mm.stride(0), mm.stride(1), *partial.stride(),
            n, m, i, p, j, k, l, s,
            (engine.HGS-engine.LGS)*engine.vread*k, float(engine.rdac-1), float(engine.radc-1),
            engine.vread, float(2**(x.total_bits-1)-1), float(2**(mat.total_bits-1)-1),
            2 if engine.execution_mode == "speed" else 0, "ieee",
            x.is_uniform_1bit_slices, block_r, 32, block_k, nr, nl, m, 1,
            COUNT_ADC=engine.adc_clip, ADC_MAX_CODE=engine.radc-1,
            REFERENCE=reference, APPLY_REFERENCE=reference is not None,
            BF16_CURRENT=engine.execution_mode == "speed", CLAMP_NORMALIZED=True,
            INTEGER_ADC=True, WEIGHT_SLICE_MAX=mat.sliced_max_weights,
            COLUMN_SCALES=mm.shape[-1] != 1, MM_COLUMN_STRIDE=mm.stride(-1),
            num_warps=4, num_stages=2 if tuned else 3, enable_fp_fusion=False,
        )
    return partial.sum(0)
