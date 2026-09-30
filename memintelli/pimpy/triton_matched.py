"""Independent matched-rounding restoration and fixed partial reduction."""
import math

import torch
import triton
import triton.language as tl

from .triton_bf16_kernel import _bf16_partial_kernel


@triton.jit
def _restore(
    idx, noise, out, offset_base,
    TOTAL: tl.constexpr, P: tl.constexpr, S: tl.constexpr,
    D0: tl.constexpr, D1: tl.constexpr, D2: tl.constexpr,
    D3: tl.constexpr, D4: tl.constexpr,
    LGS: tl.constexpr, QG: tl.constexpr, SIGMA: tl.constexpr,
    PACKED: tl.constexpr, SUPPLIED: tl.constexpr, BLOCK: tl.constexpr,
):
    pos = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = pos < TOTAL
    col = pos % 64
    row = pos // 64 % 64
    sl = pos // 4096 % S
    tile_p = pos // (4096 * S) % P
    tile_m = (pos // (4096 * S * P)).to(tl.int64)
    if PACKED:
        address = tile_m * D0 + tile_p * D1 + row * D3 + (col // 2) * D4
        byte = tl.load(idx + address, mask=valid, other=0).to(tl.int32)
        index = (((byte >> ((col % 2) * 4 + sl)) & 1) * 15).to(tl.float32)
    else:
        address = tile_m * D0 + tile_p * D1 + sl * D2 + row * D3 + col * D4
        index = tl.load(idx + address, mask=valid, other=0).to(tl.float32)
    shifted = (index * QG).to(tl.bfloat16).to(tl.float32)
    if SIGMA > 0:
        absolute = (shifted + LGS).to(tl.bfloat16).to(tl.float32)
        if SUPPLIED:
            normal = tl.load(noise + pos, mask=valid, other=0)
        else:
            normal = tl.randn(12345, offset_base + pos)
        multiplier = tl.exp(normal * SIGMA)
        noisy = (absolute * multiplier).to(tl.bfloat16).to(tl.float32)
        shifted = noisy - LGS
    tl.store(out + pos, shifted, mask=valid)


def restore_conductance(state, *, packed=False, sigma=0.0, seed=12345,
                        offset=0, block=128, supplied_noise=None, slices=4):
    """Restore one output slab. Supplied FP32 normals support paired diagnostics."""
    if slices not in (4, 6) or (packed and slices != 4):
        raise ValueError("Unsupported slice count or compact encoding")
    shape_tail = (1, 64, 32) if packed else (slices, 64, 64)
    if (not state.is_cuda or state.dtype != torch.uint8 or state.ndim != 5
            or tuple(state.shape[2:]) != shape_tail or min(state.shape[:2]) <= 0):
        raise ValueError("Unsupported CUDA mapped-state tensor")
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError("sigma must be finite and nonnegative")
    shape = (*state.shape[:2], slices, 64, 64)
    if supplied_noise is not None and (
        supplied_noise.shape != shape or supplied_noise.device != state.device
        or supplied_noise.dtype != torch.float32 or not supplied_noise.is_contiguous()
    ):
        raise ValueError("Noise must be contiguous FP32 with the logical conductance shape")
    if block not in (128, 256, 512):
        raise ValueError("restore block must be 128, 256 or 512")
    if sigma > 0:
        offset = (int(offset) + ((int(seed) & 0x7fffffff) % 65521) * 131071) & 0x7fffffff
    out = torch.empty(shape, dtype=torch.bfloat16, device=state.device)
    with torch.cuda.device(state.device):
        _restore[(triton.cdiv(out.numel(), block),)](
            state, state if supplied_noise is None else supplied_noise, out, int(offset),
            out.numel(), state.shape[1], slices, *state.stride(), 1e-7, (1e-5 - 1e-7) / 15,
            float(sigma), packed, supplied_noise is not None, block,
            num_warps=4, enable_fp_fusion=False,
        )
    return out


def accumulate(x, conductance, slice_scale, mat_max, adc_ref, radc, *, vread=0.2,
               count_adc=False, reference=None):
    """Fixed [input_tile,token,output] partial layout; no atomics."""
    xs = x.sliced_data.contiguous()
    g = conductance.contiguous()
    scales = slice_scale.contiguous()
    xm = x.max_data.contiguous()
    mm = mat_max.contiguous()
    xmax = x.sliced_max_weights.contiguous()
    n, m, i, j, k = xs.shape
    gm, p, s, gk, l = g.shape
    if (gm, gk, j, k, l) != (m, k, 1, 64, 64) or i not in (4, 6) or s not in (4, 6):
        raise ValueError("Unsupported exact-profile sliced shape")
    if scales.shape != (i, s):
        raise ValueError("Slice reconstruction scales do not match operands")
    if reference is not None:
        if not count_adc or reference.shape != (m, p, s, 2) or reference.device != xs.device:
            raise ValueError("Reference tracking requires matching count-ADC array coefficients")
        reference = reference.contiguous()
    partial = torch.empty((m, n * j, p * l), dtype=torch.float32, device=xs.device)
    nr, nl = triton.cdiv(n * j, 64), triton.cdiv(l, 32)
    with torch.cuda.device(xs.device):
        _bf16_partial_kernel[(nr * m * p * nl,)](
            xs, g, xmax, scales, xm, mm, partial,
            *xs.stride(), *g.stride(), *scales.stride(),
            xm.stride(0), xm.stride(1), mm.stride(0), mm.stride(1),
            *partial.stride(), n, m, i, p, j, k, l, s,
            float(adc_ref), 1.0, float(radc - 1), float(vread),
            float(2**(i-1)-1), float(2**(s-1)-1), 2, "ieee",
            True, 64, 32, 64, nr, nl, m, 1,
            COUNT_ADC=count_adc, ADC_MAX_CODE=int(radc - 1),
            REFERENCE=reference, APPLY_REFERENCE=reference is not None,
            num_warps=4, num_stages=2, enable_fp_fusion=False,
        )
        output = torch.empty((n * j, p * l), dtype=torch.float32, device=xs.device)
        torch.sum(partial, dim=0, out=output)
    return output
