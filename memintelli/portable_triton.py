"""Optional Triton backend. Imported only when selected or attempted."""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from .pimpy.triton_bf16_kernel import _bf16_partial_kernel
from .pimpy.triton_fast_accumulate import _round_even


@triton.jit
def _device_hash(value):
    value = value.to(tl.uint32)
    value = (value ^ (value >> 16)) * 0x7feb352d
    value = (value ^ (value >> 15)) * 0x846ca68b
    return value ^ (value >> 16)


@triton.jit
def _sqrt_rn(value):
    # The default libdevice lowering may choose approximate sqrt. Match Torch's
    # correctly rounded square root before the Box-Muller sample is multiplied.
    return tl.inline_asm_elementwise(
        "sqrt.rn.f32 $0, $1;", constraints="=f,f", args=[value],
        dtype=tl.float32, is_pure=True, pack=1,
    )


@triton.jit
def _device_normal(address, seed):
    first = _device_hash(address.to(tl.uint32) ^ seed.to(tl.uint32))
    second = _device_hash(first ^ 0xa511e9b3)
    u = ((first >> 8).to(tl.float32) + 0.5) * (1.0 / 16777216.0)
    v = ((second >> 8).to(tl.float32) + 0.5) * (1.0 / 16777216.0)
    return _sqrt_rn(-2.0 * libdevice.log(u)) * libdevice.cos(6.2831854820251465 * v)


@triton.jit
def _restore_kernel(INDICES, NOMINAL, WRITE_SIGMA, READ_SIGMA, DRIFT, OUTPUT,
                    COUNT: tl.constexpr, SOURCE_COLUMNS: tl.constexpr,
                    COLUMNS: tl.constexpr, START, DEVICES_PER_TILE: tl.constexpr,
                    WRITE_SEED, READ_SEED, LGS: tl.constexpr,
                    WRITE: tl.constexpr, READ: tl.constexpr, RETENTION: tl.constexpr,
                    BLOCK: tl.constexpr):
    index = tl.program_id(0).to(tl.int64)*BLOCK + tl.arange(0, BLOCK)
    block = index // (COLUMNS*DEVICES_PER_TILE)
    within = index % (COLUMNS*DEVICES_PER_TILE)
    address = (block*SOURCE_COLUMNS + START)*DEVICES_PER_TILE + within
    state = tl.load(INDICES + address, index < COUNT, other=0).to(tl.int32)
    g = tl.load(NOMINAL + state)
    if WRITE:
        g = g * libdevice.exp(_device_normal(address, WRITE_SEED) * tl.load(WRITE_SIGMA + state))
    if RETENTION:
        g = g * tl.load(DRIFT + state)
    if READ:
        g = g * libdevice.exp(_device_normal(address, READ_SEED) * tl.load(READ_SIGMA + state))
    g = g.to(tl.bfloat16, fp_downcast_rounding="rtne").to(tl.float32)
    g = (g - LGS).to(tl.bfloat16, fp_downcast_rounding="rtne")
    tl.store(OUTPUT + index, g, index < COUNT)


def restore_conductance(mat, engine, start, end, read_epoch):
    """Restore one logical device sample in native array layout, without FP32 temporaries."""
    if engine.execution_mode != "speed" or not mat.G_indices.is_contiguous():
        return None
    m, p, s, k, l = mat.logical_index_shape
    output = torch.empty((m, end-start, s, k, l), device=engine.device, dtype=torch.bfloat16)
    layer_key = engine.seed + mat.simulation_layer_id*1000003
    with torch.cuda.device(engine.device):
        _restore_kernel[(triton.cdiv(output.numel(), 256),)](
            mat.G_indices, engine._nominal_conductance, engine.write_sigmas,
            engine.read_sigmas, engine._drift_factors(), output,
            output.numel(), p, end-start, start, s*k*l,
            (layer_key + mat.program_epoch*31337) & 0xffffffff,
            (layer_key + 0x13579bdf + read_epoch*104729) & 0xffffffff,
            engine.LGS, engine._write_active, engine._read_active, engine._drift_active,
            256, num_warps=4, enable_fp_fusion=False,
        )
    return output


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
