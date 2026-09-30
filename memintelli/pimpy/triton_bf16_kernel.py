"""Static BF16-current kernel derived from the v3 partial-reduction path.

Each slice-pair keeps its own ADC. The caller disables floating-point fusion.
"""
import triton
import triton.language as tl
from .triton_fast_accumulate import _round_even


@triton.jit
def _adc_round_even(value):
    half = _round_even(value * 2.0) * 0.5
    tolerance = tl.minimum(tl.maximum(tl.abs(value), 1.0) * 4.76837158203125e-7, 1e-4)
    value = tl.where(tl.abs(value-half) <= tolerance, half, value)
    return _round_even(value)


@triton.jit
def _bf16_partial_kernel(
    x_sliced,
    g0,
    x_slice_max,
    scale,
    x_max,
    mat_max,
    partial,
    x_s0: tl.constexpr,
    x_s1: tl.constexpr,
    x_s2: tl.constexpr,
    x_s3: tl.constexpr,
    x_s4: tl.constexpr,
    g_s0: tl.constexpr,
    g_s1: tl.constexpr,
    g_s2: tl.constexpr,
    g_s3: tl.constexpr,
    g_s4: tl.constexpr,
    scale_s0: tl.constexpr,
    scale_s1: tl.constexpr,
    xm_s0: tl.constexpr,
    xm_s1: tl.constexpr,
    mm_s0: tl.constexpr,
    mm_s1: tl.constexpr,
    part_s0: tl.constexpr,
    part_s1: tl.constexpr,
    part_s2: tl.constexpr,
    N: tl.constexpr,
    M: tl.constexpr,
    I: tl.constexpr,
    P: tl.constexpr,
    J: tl.constexpr,
    K: tl.constexpr,
    L: tl.constexpr,
    S: tl.constexpr,
    ADC_REF: tl.constexpr,
    RDAC_SCALE: tl.constexpr,
    RADC_SCALE: tl.constexpr,
    VREAD: tl.constexpr,
    X_QMAX: tl.constexpr,
    MAT_QMAX: tl.constexpr,
    DOT_DTYPE: tl.constexpr,
    INPUT_PRECISION: tl.constexpr,
    BINARY_INPUT_SLICES: tl.constexpr,
    BLOCK_R: tl.constexpr,
    BLOCK_L: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_R_BLOCKS: tl.constexpr,
    NUM_L_BLOCKS: tl.constexpr,
    NUM_M_GROUPS: tl.constexpr,
    GROUP_M_TILES: tl.constexpr,
    ADC_TRACE=None,
    TRACE: tl.constexpr = False,
    COUNT_ADC: tl.constexpr = False,
    ADC_MAX_CODE: tl.constexpr = 63,
    TRACE_SIGNAL: tl.constexpr = False,
    REFERENCE=None,
    APPLY_REFERENCE: tl.constexpr = False,
    BF16_CURRENT: tl.constexpr = True,
    CLAMP_NORMALIZED: tl.constexpr = False,
    INTEGER_ADC: tl.constexpr = False,
    WEIGHT_SLICE_MAX=None,
    COLUMN_SCALES: tl.constexpr = False,
    MM_COLUMN_STRIDE: tl.constexpr = 0,
):
    pid = tl.program_id(0)
    pid_r = pid % NUM_R_BLOCKS
    pid_rest = pid // NUM_R_BLOCKS
    pid_mg = pid_rest % NUM_M_GROUPS
    pid_pl = pid_rest // NUM_M_GROUPS
    pid_p = pid_pl // NUM_L_BLOCKS
    pid_l = pid_pl - pid_p * NUM_L_BLOCKS

    offs_r = pid_r * BLOCK_R + tl.arange(0, BLOCK_R)
    offs_l = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
    offs_k = tl.arange(0, BLOCK_K)
    offs_n = offs_r // J
    offs_j = offs_r - offs_n * J
    mask_r = offs_r < (N * J)
    mask_l = offs_l < L

    group_total = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
    base_m = pid_mg * GROUP_M_TILES

    for gm_i in tl.range(0, GROUP_M_TILES, 1, loop_unroll_factor=1):
        m = base_m + gm_i
        valid_m = m < M
        tile = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
        x_tile_max = tl.load(
            x_max + offs_n * xm_s0 + m * xm_s1,
            mask=mask_r & valid_m,
            other=0.0,
        ).to(tl.float32)
        if COLUMN_SCALES:
            mat_tile_max = tl.load(
                mat_max + m * mm_s0 + pid_p * mm_s1 + offs_l * MM_COLUMN_STRIDE,
                mask=valid_m & (pid_p < P) & mask_l, other=0.0,
            )[None, :].to(tl.float32)
        else:
            mat_tile_max = tl.load(
                mat_max + m * mm_s0 + pid_p * mm_s1,
                mask=valid_m & (pid_p < P), other=0.0,
            ).to(tl.float32)

        for i in tl.static_range(0, I):
            if not BINARY_INPUT_SLICES:
                xmax_i = tl.load(x_slice_max + i).to(tl.float32)
            else:
                xmax_i = 1.0
            for s in tl.static_range(0, S):
                cur = tl.zeros((BLOCK_R, BLOCK_L), dtype=tl.float32)
                if APPLY_REFERENCE:
                    population = tl.zeros((BLOCK_R,), dtype=tl.float32)
                for k0 in tl.static_range(0, K, BLOCK_K):
                    k = k0 + offs_k
                    mask_k = k < K
                    x_raw = tl.load(
                        x_sliced
                        + offs_n[:, None] * x_s0
                        + m * x_s1
                        + i * x_s2
                        + offs_j[:, None] * x_s3
                        + k[None, :] * x_s4,
                        mask=valid_m & mask_r[:, None] & mask_k[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    if APPLY_REFERENCE:
                        population += tl.sum(x_raw, axis=1)
                    if BINARY_INPUT_SLICES:
                        v = x_raw * VREAD
                    else:
                        v = _round_even(x_raw / xmax_i * RDAC_SCALE) * (VREAD / RDAC_SCALE)
                    w0 = tl.load(
                        g0
                        + m * g_s0
                        + pid_p * g_s1
                        + s * g_s2
                        + k[:, None] * g_s3
                        + offs_l[None, :] * g_s4,
                        mask=valid_m & mask_k[:, None] & mask_l[None, :],
                        other=0.0,
                    )
                    if DOT_DTYPE == 1:
                        v = v.to(tl.float16)
                        w0 = w0.to(tl.float16)
                    elif DOT_DTYPE == 2:
                        v = v.to(tl.bfloat16)
                        w0 = w0.to(tl.bfloat16)
                    cur += tl.dot(v, w0, input_precision=INPUT_PRECISION)

                if BF16_CURRENT:
                    cur = cur.to(tl.bfloat16).to(tl.float32)
                if COUNT_ADC:
                    if INTEGER_ADC:
                        wmax_s = tl.load(WEIGHT_SLICE_MAX + s).to(tl.float32)
                        maximum = K * xmax_i * wmax_s
                        full_range = tl.exp2(tl.maximum(1.0, tl.ceil(tl.log2(maximum))))
                        adc_step = full_range / (ADC_MAX_CODE + 1)
                        limit = tl.minimum(maximum, full_range - 1.0)
                        highest_code = tl.minimum(ADC_MAX_CODE, tl.floor(limit / adc_step))
                    else:
                        maximum = K
                        adc_step = 1.0
                        highest_code = ADC_MAX_CODE
                    signal = cur / ADC_REF * maximum
                    if APPLY_REFERENCE:
                        address = ((m * P + pid_p) * S + s) * 2
                        gain = tl.load(REFERENCE + address, mask=valid_m, other=1.0)
                        baseline = tl.load(REFERENCE + address + 1, mask=valid_m, other=0.0)
                        signal = tl.div_rn(signal - baseline * population[:, None], gain)
                    if INTEGER_ADC and not BF16_CURRENT:
                        code = _adc_round_even(signal / adc_step)
                    else:
                        code = _round_even(signal / adc_step)
                    code = tl.minimum(tl.maximum(code, 0.0), highest_code)
                    if INTEGER_ADC:
                        q = code * adc_step
                    else:
                        q = code * adc_step / maximum
                else:
                    if INTEGER_ADC and not BF16_CURRENT:
                        q = _adc_round_even(cur / ADC_REF * RADC_SCALE) / RADC_SCALE
                    else:
                        q = _round_even(cur / ADC_REF * RADC_SCALE) / RADC_SCALE
                    if CLAMP_NORMALIZED:
                        q = tl.minimum(tl.maximum(q, 0.0), 1.0)
                if TRACE:
                    trace_address = (((m * P + pid_p) * I + i) * S + s) * N * L
                    traced = cur / ADC_REF * K if TRACE_SIGNAL else q * K
                    tl.store(ADC_TRACE + trace_address + offs_n[:, None] * L + offs_l[None, :],
                             traced, mask=valid_m & mask_r[:, None] & mask_l[None, :])
                scale_is = tl.load(scale + i * scale_s0 + s * scale_s1)
                tile += q * scale_is

        bm = x_tile_max[:, None] * mat_tile_max
        tile = tile * bm
        tile = tile / X_QMAX
        tile = tile / MAT_QMAX
        group_total += tile

    local_cols = pid_p * L + offs_l
    tl.store(
        partial + pid_mg.to(tl.int64) * part_s0 + offs_r[:, None] * part_s1 + local_cols[None, :] * part_s2,
        group_total,
        mask=mask_r[:, None] & mask_l[None, :],
    )
