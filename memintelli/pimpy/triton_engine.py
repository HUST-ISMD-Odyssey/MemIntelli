"""Restricted, fail-fast v3.1 inference profile; legacy engines stay available."""
import math

import torch

from .compact_state import pack_indices
from .memmat_tensor_multimode import DPETensorMultiMode


class TritonEngine(DPETensorMultiMode):
    """Mode0, configurable one-bit slices, BF16 current, fixed reduction.

    ``compact`` changes mapped-state storage only. Output chunking is explicit:
    changing it can change the reduction tree and the read-noise sample layout.
    This is an inference-only API, not a universal bitwise-equivalence guarantee.
    """

    use_multimode_sliced_data = True
    require_bf16_input = True

    def __init__(self, *, device="cuda:0", read_variation=0.05,
                 compact=True, output_chunk_tiles=2374, read_variation_seed=None,
                 activation_bits=4, weight_bits=4):
        for bits in (activation_bits, weight_bits):
            if isinstance(bits, bool) or not isinstance(bits, int) or bits not in (4, 6):
                raise ValueError("Supported activation and weight widths are 4 or 6 bits")
        if compact and weight_bits != 4:
            raise ValueError("Compact mapped storage supports W4 only; use compact=False for W6")
        self.activation_bits, self.weight_bits = activation_bits, weight_bits
        device = torch.device(device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("TritonEngine requires a CUDA GPU and a CUDA PyTorch build")
        from .triton_fast_accumulate import is_triton_fast_accumulate_available
        if not is_triton_fast_accumulate_available():
            raise RuntimeError("Triton is required; install the triton extra on Linux")
        if not math.isfinite(read_variation) or read_variation < 0:
            raise ValueError("read_variation must be finite and nonnegative")
        if isinstance(output_chunk_tiles, bool) or int(output_chunk_tiles) != output_chunk_tiles or output_chunk_tiles < 1:
            raise ValueError("output_chunk_tiles must be a positive integer")
        super().__init__(
            device=device, mode=0, write_variation=0, read_variation=read_variation,
            vnoise=0, rate_stuck_HGS=0, rate_stuck_LGS=0,
            rdac=2, radc=256, conductance_dtype=torch.bfloat16,
            compute_dtype=torch.bfloat16, mode0_vmm_compute_dtype="bf16",
            fast_inference=True, fast_inference_backend="triton",
            triton_fuse_restored_input_slices=True,
            triton_direct_final_exact_reduce=True, triton_gidx_direct_final_output=False,
            triton_block_r=64, triton_block_l=32, triton_block_k=64,
            triton_output_chunk_limit=int(output_chunk_tiles),
            inference_input_chunk_size=0, read_variation_seed=read_variation_seed,
        )
        self.compact = bool(compact)
        self.execution_counts = {"mapped": 0, "restores": 0, "triton_partials": 0,
                                 "forwards": 0, "fallbacks": 0}

    def _validate_format(self, data, *, weight):
        self._validate_sliced_data_mode(data, "weight" if weight else "input")
        tile = (64, 64) if weight else (1, 64)
        bits = self.weight_bits if weight else self.activation_bits
        if (not data.inference or data.bw_e is not None
                or tuple(data.paral_size) != tile or tuple(data.quant_gran) != tile
                or data.total_bits != bits or len(data.slice_method) != bits
                or not data.is_uniform_1bit_slices):
            raise ValueError("TritonEngine requires configured one-bit slices and tile-local quantization")

    def _prepare_weight_conductance(self, mat):
        self._validate_format(mat, weight=True)
        super()._prepare_weight_conductance(mat)
        mat.compress_G(self)
        mat.compact_mapped_state = False
        mat.logical_index_shape = tuple(mat.G_indices.shape)
        if self.compact:
            mat.G_indices = pack_indices(mat.G_indices)
            mat.compact_mapped_state = True
        self.execution_counts["mapped"] += 1

    def MapReduceDot(self, x, mat):
        from .triton_matched import accumulate, restore_conductance
        self._validate_format(x, weight=False)
        self._validate_format(mat, weight=True)
        if (len(x.shape) != 2 or len(mat.shape) != 2
                or x.shape[-1] != mat.shape[0] or x.shape[0] < 1):
            raise ValueError("TritonEngine expects nonempty compatible 2-D inputs")
        if (not x.sliced_data.is_cuda or not mat.G_indices.is_cuda
                or mat.G_indices.device != x.sliced_data.device):
            raise ValueError("Mapped state and inputs must be on the same CUDA device")
        m, p, s, k, l = mat.logical_index_shape
        width = min(p, self.triton_output_chunk_limit, max(1, 2_000_000_000 // (m * s * k * l)))
        adc_ref = (self.HGS - self.LGS) * self.vread * k
        scale_base = adc_ref / self.Q_G / self.vread / (self.g_level - 1)
        scale = (
            x.sliced_max_weights.view(-1, 1) * x.sliced_weights.view(-1, 1)
            * mat.sliced_max_weights.view(1, -1) * mat.sliced_weights.view(1, -1)
            * scale_base
        )
        epoch = self._read_noise_forward_offset_counter * math.prod(mat.logical_index_shape)
        if self._has_read_noise:
            self._read_noise_forward_offset_counter += 1
        chunks = []
        for start in range(0, p, width):
            end = min(start + width, p)
            state = mat.G_indices[:, start:end]
            block = self._select_triton_gidx_restore_block(state)
            g = restore_conductance(
                state, packed=mat.compact_mapped_state,
                sigma=float(self._rv_sigma if self._has_read_noise else 0),
                seed=self._read_noise_seed_base,
                offset=epoch + start * m * s * k * l, block=block,
                slices=s,
            )
            self.execution_counts["restores"] += 1
            chunk = accumulate(x, g, scale, mat.max_data[:, start:end], adc_ref, self.radc)
            self.execution_counts["triton_partials"] += 1
            chunks.append(chunk)
            del g
        self.execution_counts["forwards"] += 1
        output = chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=1)
        return output[:, :mat.shape[1]]
