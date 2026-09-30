"""Portable bit-sliced inference with a shared device model and two backends."""
import math
import warnings
import copy
from numbers import Real

import torch

from .pimpy.data_formats_multimode import SlicedDataMultiMode
from .pimpy.memmat_tensor_multimode import DPETensorMultiMode


def _integer(value, name, minimum=0, maximum=None):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or int(value) != value
            or value < minimum or (maximum is not None and value > maximum)):
        bound = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be an integer {bound}")
    return int(value)


def _real(value, name, minimum=0, positive=False):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < minimum or (positive and value == minimum)):
        raise ValueError(f"{name} must be finite and {'>' if positive else '>='} {minimum}")
    return float(value)


def _pair(value, name):
    try:
        values = tuple(value)
    except TypeError:
        raise ValueError(f"{name} must contain two positive integers") from None
    if len(values) != 2:
        raise ValueError(f"{name} must contain two positive integers")
    return tuple(_integer(x, name, 1) for x in values)


def _slices(value, bits, name):
    if value is None:
        return (1,) * bits
    try:
        result = tuple(_integer(x, name, 1, 8) for x in value)
    except TypeError:
        raise ValueError(f"{name} must be a nonempty sequence of slice widths") from None
    if not result or result[0] != 1 or sum(result) != bits:
        raise ValueError(f"{name}: sign bit first, widths in [1, 8], sum must equal {bits}")
    return result


def _levels(value, count, name, device):
    if isinstance(value, dict):
        parsed = {}
        for key, item in value.items():
            try:
                key = int(key) if isinstance(key, str) else _integer(key, name, 0, count-1)
            except (ValueError, TypeError):
                raise ValueError(f"{name} keys must be integer state indices") from None
            if key in parsed:
                raise ValueError(f"{name} has duplicate state index {key}")
            parsed[key] = item
        value = parsed
        if set(value) != set(range(count)):
            raise ValueError(f"{name} must specify every state from 0 to {count - 1}")
        value = [value[k] for k in range(count)]
    elif isinstance(value, Real):
        value = [value] * count
    try:
        result = torch.as_tensor(value, device=device, dtype=torch.float32)
    except (ValueError, TypeError):
        raise ValueError(f"{name} must be a scalar, state list or complete state dictionary") from None
    if result.shape != (count,) or not torch.isfinite(result).all() or (result < 0).any():
        raise ValueError(f"{name} must contain {count} finite nonnegative values")
    return result


def _hash(value):
    value = value & 0xffffffff
    value = ((value ^ (value >> 16)) * 0x7feb352d) & 0xffffffff
    value = ((value ^ (value >> 15)) * 0x846ca68b) & 0xffffffff
    return value ^ (value >> 16)


def _normal(address, seed):
    # Logical device addresses make samples independent of output chunk size.
    first = _hash(address ^ (int(seed) & 0xffffffff))
    second = _hash(first ^ 0xa511e9b3)
    u = ((first >> 8).float() + 0.5) / 16777216
    v = ((second >> 8).float() + 0.5) / 16777216
    return torch.sqrt(-2 * torch.log(u)) * torch.cos(2 * math.pi * v)


class SimulationEngine(DPETensorMultiMode):
    """Signed INT bit slicing. ``auto`` tries Triton, then visibly falls back.

    A physical array has ``weight_paral_size`` rows/columns. Row scale groups
    span complete array rows; column groups may be narrower than one array.
    Read noise is refreshed per mapped-layer call and shared over that call's
    batch and activation slices.
    """

    use_multimode_sliced_data = True
    require_bf16_input = False
    supports_column_scales = True

    def __init__(
        self, *, backend="auto", mode="speed", device=None,
        activation_bits=6, weight_bits=6, input_slice=None, weight_slice=None,
        weight_paral_size=(64, 64), input_quant_gran=None, weight_quant_gran=None,
        HGS=1e-5, LGS=1e-7, g_level=16, vread=0.2,
        adc_bits=6, dac_bits=1, adc_clip=True,
        write_variation=0.0, read_variation=0.0,
        drift_coefficient=0.0, drift_time=0.0, drift_reference_time=1.0,
        seed=42, program_epoch=0, output_chunk_tiles=8, input_chunk_rows=256,
        torch_fuse_adc=True,
    ):
        if backend not in ("auto", "triton", "torch"):
            raise ValueError("backend must be auto, triton or torch")
        if mode not in ("speed", "accurate"):
            raise ValueError("mode must be speed or accurate")
        device = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
        if device.type not in ("cpu", "cuda"):
            raise ValueError("Supported devices are CPU and CUDA")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        if device.type == "cpu":
            device = torch.device("cpu")
        elif device.index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {device.index} is unavailable")
        activation_bits = _integer(activation_bits, "activation_bits", 2, 16)
        weight_bits = _integer(weight_bits, "weight_bits", 2, 16)
        adc_bits = _integer(adc_bits, "adc_bits", 1, 16)
        dac_bits = _integer(dac_bits, "dac_bits", 1, 16)
        self.input_slice = _slices(input_slice, activation_bits, "input_slice")
        self.weight_slice = _slices(weight_slice, weight_bits, "weight_slice")
        g_level = _integer(g_level, "g_level", 2, 256)
        HGS, LGS = _real(HGS, "HGS", positive=True), _real(LGS, "LGS", positive=True)
        vread = _real(vread, "vread", positive=True)
        if LGS >= HGS:
            raise ValueError("Require LGS < HGS")
        if not isinstance(adc_clip, bool):
            raise ValueError("adc_clip must be a boolean")
        if not isinstance(torch_fuse_adc, bool):
            raise ValueError("torch_fuse_adc must be a boolean")
        if 2**max(self.weight_slice) > g_level or 2**max(self.input_slice) > 2**dac_bits:
            raise ValueError("Conductance/DAC levels must cover the largest weight/input slice")
        self.weight_paral_size = _pair(weight_paral_size, "weight_paral_size")
        self.input_paral_size = (1, self.weight_paral_size[0])
        self.input_quant_gran = _pair(
            self.input_paral_size if input_quant_gran is None else input_quant_gran, "input_quant_gran")
        self.weight_quant_gran = _pair(
            self.weight_paral_size if weight_quant_gran is None else weight_quant_gran, "weight_quant_gran")
        if self.input_quant_gran[0] != 1:
            raise ValueError("Input quantization uses one independent scale group per sample row")
        if self.input_quant_gran[1] % self.weight_paral_size[0]:
            raise ValueError("Input scale groups must span complete array rows")
        if self.weight_quant_gran[0] % self.weight_paral_size[0]:
            raise ValueError("Weight scale groups must span complete array rows; column groups may be smaller")
        output_chunk_tiles = _integer(output_chunk_tiles, "output_chunk_tiles", 1)
        input_chunk_rows = _integer(input_chunk_rows, "input_chunk_rows", 1)
        drift_time = _real(drift_time, "drift_time")
        drift_reference_time = _real(drift_reference_time, "drift_reference_time", positive=True)
        seed = _integer(seed, "seed", 0, 2**32-1)
        program_epoch = _integer(program_epoch, "program_epoch")
        numeric_values = (HGS, LGS, vread, (HGS-LGS)/(g_level-1),
                          (HGS-LGS)*vread*self.weight_paral_size[0])
        precision = torch.finfo(torch.float32)
        if any(not precision.tiny <= value <= precision.max for value in numeric_values):
            raise ValueError("Conductance, voltage, conductance step and ADC reference must fit normal float32")
        super().__init__(
            device=device, mode=0, HGS=HGS, LGS=LGS, g_level=int(g_level), vread=vread,
            write_variation=0, read_variation=0, vnoise=0,
            rate_stuck_HGS=0, rate_stuck_LGS=0, rdac=2**dac_bits, radc=2**adc_bits,
            conductance_dtype=torch.float32, compute_dtype=torch.float32,
            inference_input_chunk_size=0, fast_inference_backend="torch",
            triton_fuse_activation_slices=False,
        )
        self.backend_requested = backend
        self.backend_used = None
        self.execution_mode = mode
        self.activation_bits, self.weight_bits = activation_bits, weight_bits
        self.adc_clip, self.adc_bits, self.dac_bits = adc_clip, adc_bits, dac_bits
        self.torch_fuse_adc = torch_fuse_adc
        self._torch_fusion_status = "not_used" if torch_fuse_adc else "disabled"
        self._torch_fusion_error = None
        self.seed, self.program_epoch = int(seed), int(program_epoch)
        self.output_chunk_tiles, self.input_chunk_rows = int(output_chunk_tiles), int(input_chunk_rows)
        self.write_sigmas = _levels(write_variation, g_level, "write_variation", device)
        self.read_sigmas = _levels(read_variation, g_level, "read_variation", device)
        self.drift_exponents = _levels(drift_coefficient, g_level, "drift_coefficient", device)
        self.retention_time, self.retention_reference = float(drift_time), float(drift_reference_time)
        self._write_active = bool((self.write_sigmas > 0).any())
        self._read_active = bool((self.read_sigmas > 0).any())
        self._drift_active = drift_time > 0 and bool((self.drift_exponents > 0).any())
        self._next_layer_id = 0
        self._mapping_token = object()
        self.layer_calls = {}
        self._fallback_reason = None
        self.execution_counts = {"mapped": 0, "triton": 0, "torch": 0, "fallbacks": 0}
        self._input_template = SlicedDataMultiMode(
            torch.tensor(self.input_slice), inference=True, device=self.device,
            paral_size=self.input_paral_size, quant_gran=self.input_quant_gran,
        )
        widths = tuple(reversed(self.input_slice))
        shifts, offset = [], 0
        for width in widths:
            shifts.append(offset)
            offset += width
        self._input_shifts = torch.tensor(shifts, device=device, dtype=torch.int32)[None, None, :, None]
        self._input_masks = torch.tensor([2**b-1 for b in widths], device=device,
                                        dtype=torch.int32)[None, None, :, None]
        indices = torch.arange(g_level, device=device, dtype=torch.float32)
        nominal = self._round_internal(self._round_internal(indices*self.Q_G) + self.LGS)
        self._nominal_shifted = self._round_internal(nominal-self.LGS)
        weight_meta = SlicedDataMultiMode(torch.tensor(self.weight_slice), device=device)
        self._slice_scales = (self._input_template.sliced_weights[:, None]
                              * weight_meta.sliced_weights[None, :])
        if not adc_clip:
            self._slice_scales = (self._slice_scales
                                  * self._input_template.sliced_max_weights[:, None]
                                  * weight_meta.sliced_max_weights[None, :]
                                  * self.weight_paral_size[0])
        if backend != "torch":
            try:
                if device.type != "cuda":
                    raise RuntimeError("Triton needs a CUDA device")
                from .portable_triton import accumulate, slice_input
                self._triton_accumulate = accumulate
                self._triton_slice_input = slice_input
            except Exception as exc:
                self._fallback(exc)

    def _fallback(self, exc):
        if self.backend_requested == "triton":
            raise RuntimeError(f"Triton backend unavailable or failed: {exc}") from exc
        if self._fallback_reason is None:
            self._fallback_reason = str(exc)
            self.execution_counts["fallbacks"] += 1
            warnings.warn(f"Triton unavailable/failed; using PyTorch: {exc}", RuntimeWarning, stacklevel=3)
        self.backend_used = "torch"

    def _prepare_weight_conductance(self, mat):
        if mat.bw_e is not None or mat.mode != 0:
            raise ValueError("SimulationEngine supports signed integer slicing (bw_e=None, mode=0)")
        rows = math.ceil(mat.shape[0] / self.weight_paral_size[0])
        columns = math.ceil(mat.shape[1] / self.weight_paral_size[1])
        levels = mat.sliced_data[:rows, :columns].float() / mat.sliced_max_weights.view(1, 1, -1, 1, 1)
        mat.max_data = mat.max_data[:rows, :columns].contiguous()
        mat.G_indices = torch.round(levels * (self.g_level - 1)).to(torch.uint8)
        mat.G = None
        mat.G_is_compressed, mat.G_index_dtype = True, torch.uint8
        mat.logical_index_shape = tuple(mat.G_indices.shape)
        if not hasattr(mat, "simulation_layer_id"):
            mat.simulation_layer_id = self._next_layer_id
            self._next_layer_id += 1
            self.execution_counts["mapped"] += 1
        mat.program_epoch = self.program_epoch
        mat._mapping_token = self._mapping_token

    def reset_read_sequence(self):
        """Replay read noise without reprogramming weights."""
        self.layer_calls.clear()

    def _round_internal(self, tensor):
        return tensor.to(torch.bfloat16).float() if self.execution_mode == "speed" else tensor

    def restore(self, mat, start=0, end=None, *, read_epoch=0):
        end = mat.G_indices.shape[1] if end is None else end
        indices = mat.G_indices[:, start:end].long()
        if not (self._write_active or self._read_active or self._drift_active):
            return self._nominal_shifted[indices]
        m, p, s, k, l = mat.logical_index_shape
        address = (torch.arange(m, device=self.device)[:, None] * p
                   + torch.arange(start, end, device=self.device)[None, :])
        address = address[..., None] * (s*k*l) + torch.arange(s*k*l, device=self.device)
        address = address.reshape_as(indices)
        g = self._round_internal(indices.float() * self.Q_G)
        g = self._round_internal(g + self.LGS)
        layer_key = self.seed + mat.simulation_layer_id * 1000003
        if self._write_active:
            noise = _normal(address, layer_key + mat.program_epoch * 31337)
            g = g * torch.exp(noise * self.write_sigmas[indices])
        if self._drift_active:
            ratio = max(self.retention_time / self.retention_reference, 1.0)
            g = g * torch.pow(ratio, -self.drift_exponents[indices])
        if self._read_active:
            noise = _normal(address, layer_key + 0x13579bdf + read_epoch * 104729)
            g = g * torch.exp(noise * self.read_sigmas[indices])
        return self._round_internal(self._round_internal(g) - self.LGS)

    def _slice_input(self, inputs):
        x = copy.copy(self._input_template)
        x.shape = inputs.shape
        if self.backend_requested != "torch" and self._fallback_reason is None:
            try:
                sliced = self._triton_slice_input(inputs, self)
                if sliced is not None:
                    x.sliced_data, x.max_data = sliced
                    return x
            except Exception as exc:
                if isinstance(exc, torch.cuda.OutOfMemoryError) or any(
                    text in str(exc).lower() for text in ("illegal memory access", "device-side assert")
                ):
                    raise
                self._fallback(exc)
        rows, features = inputs.shape
        size, group = self.weight_paral_size[0], self.input_quant_gran[1]
        groups = math.ceil(features/group)
        values = torch.nn.functional.pad(inputs, (0, groups*group-features)).reshape(rows, groups, group)
        maxima = values.abs().amax(-1, keepdim=True)
        safe = torch.where(maxima > 0, maxima, 1.0)
        quantized = (values/safe*(2**(self.activation_bits-1)-1)).round().to(torch.int32)
        blocks = math.ceil(features/size)
        quantized = quantized.reshape(rows, -1, size)[:, :blocks]
        x.sliced_data = ((quantized[:, :, None, :] >> self._input_shifts)
                         & self._input_masks).to(torch.uint8).unsqueeze(-2)
        x.max_data = maxima.repeat_interleave(group//size, dim=1)[:, :blocks, :, None]
        return x

    def _accumulate(self, x, g, mat_max, mat, reference, valid_columns):
        from .portable_torch import accumulate
        if self.backend_requested != "torch" and self._fallback_reason is None:
            try:
                result = self._triton_accumulate(x, g, mat_max, mat, self, reference, valid_columns)
                self.execution_counts["triton"] += 1
                self.backend_used = "triton"
                return result[:, :valid_columns]
            except Exception as exc:
                if isinstance(exc, torch.cuda.OutOfMemoryError) or any(
                    text in str(exc).lower() for text in ("illegal memory access", "device-side assert")
                ):
                    raise
                self._fallback(exc)
        result = accumulate(x, g, mat_max, mat, self, reference, valid_columns=valid_columns)
        self.execution_counts["torch"] += 1
        self.backend_used = "torch"
        return result

    def MapReduceDot(self, x, mat):
        if getattr(mat, "_mapping_token", None) is not self._mapping_token:
            raise ValueError("Map the weight with this SimulationEngine before using it")
        if len(x.shape) != 2 or len(mat.shape) != 2 or x.shape[1] != mat.shape[0]:
            raise ValueError("Expected compatible two-dimensional sliced operands")
        if x.sliced_data.device != mat.G_indices.device or x.sliced_data.device != self.device:
            raise ValueError("Engine, weights and input must share a device")
        # Scale groups may extend beyond the matrix. They do not add physical arrays.
        expected_m = mat.logical_index_shape[0]
        x = copy.copy(x)
        x.sliced_data = x.sliced_data[:, :expected_m]
        x.max_data = x.max_data[:, :expected_m]
        if x.sliced_data.shape[1] != expected_m:
            raise ValueError("Input and weight array row dimensions do not match")
        lid = mat.simulation_layer_id
        epoch = self.layer_calls.get(lid, 0)
        self.layer_calls[lid] = epoch + 1
        parts = []
        for start in range(0, mat.G_indices.shape[1], self.output_chunk_tiles):
            end = min(start + self.output_chunk_tiles, mat.G_indices.shape[1])
            g = self.restore(mat, start, end, read_epoch=epoch)
            ref = getattr(mat, "adc_reference", None)
            if ref is not None:
                if (not self.adc_clip or max(self.input_slice + self.weight_slice) != 1
                        or ref.shape != (*mat.G_indices.shape[:3], 2)
                        or not torch.isfinite(ref).all() or (ref[..., 0] <= 0).any()):
                    raise ValueError("Invalid per-array (gain, baseline) ADC reference")
                ref = ref[:, start:end].to(self.device).contiguous()
            columns = min((end-start)*self.weight_paral_size[1],
                          mat.shape[1]-start*self.weight_paral_size[1])
            parts.append(self._accumulate(x, g, mat.max_data[:, start:end], mat, ref, columns))
        return torch.cat(parts, dim=1)[:, :mat.shape[1]]

    def map_weight(self, weight):
        """Map a [input_features, output_features] matrix once."""
        if (not isinstance(weight, torch.Tensor) or weight.layout != torch.strided
                or weight.is_complex() or weight.ndim != 2 or weight.numel() == 0):
            raise ValueError("Weights must be a nonempty, real, dense two-dimensional tensor")
        weight = weight.detach().float()
        if not torch.isfinite(weight).all():
            raise ValueError("Weights must be finite after float32 conversion")
        mat = SlicedDataMultiMode(
            torch.tensor(self.weight_slice), is_weight=True, inference=True,
            device=self.device, paral_size=self.weight_paral_size, quant_gran=self.weight_quant_gran,
        )
        mat.slice_data_imp(self, weight)
        return mat

    def matmul(self, inputs, mapped_weight):
        """Execute one logical read; row/output chunking does not resample noise."""
        if (not isinstance(inputs, torch.Tensor) or inputs.layout != torch.strided
                or inputs.is_complex()):
            raise ValueError("Inputs must be a real, dense tensor")
        if getattr(mapped_weight, "_mapping_token", None) is not self._mapping_token:
            raise ValueError("Map the weight with this SimulationEngine before using it")
        shape = inputs.shape
        if not shape or shape[-1] != mapped_weight.shape[0] or inputs.numel() == 0:
            raise ValueError("Input shape does not match mapped weight")
        flat = inputs.reshape(-1, shape[-1]).to(device=self.device, dtype=torch.float32)
        if not torch.isfinite(flat).all():
            raise ValueError("Input contains nonfinite values")
        lid = mapped_weight.simulation_layer_id
        epoch = self.layer_calls.get(lid, 0)
        result = []
        for start in range(0, len(flat), self.input_chunk_rows):
            x = self._slice_input(flat[start:start+self.input_chunk_rows])
            self.layer_calls[lid] = epoch
            result.append(self.MapReduceDot(x, mapped_weight))
        return torch.cat(result).reshape(*shape[:-1], mapped_weight.shape[1])

    def describe(self):
        return {
            "backend_requested": self.backend_requested, "backend_used": self.backend_used,
            "fallback_reason": self._fallback_reason, "mode": self.execution_mode,
            "device": str(self.device), "activation_bits": self.activation_bits,
            "weight_bits": self.weight_bits, "input_slice": self.input_slice,
            "weight_slice": self.weight_slice, "weight_paral_size": self.weight_paral_size,
            "input_paral_size": self.input_paral_size, "input_quant_gran": self.input_quant_gran,
            "weight_quant_gran": self.weight_quant_gran, "adc_bits": self.adc_bits,
            "dac_bits": self.dac_bits, "adc_clip": self.adc_clip,
            "HGS": self.HGS, "LGS": self.LGS, "g_level": self.g_level, "vread": self.vread,
            "write_variation": self.write_sigmas.cpu().tolist(),
            "read_variation": self.read_sigmas.cpu().tolist(),
            "drift_coefficient": self.drift_exponents.cpu().tolist(),
            "drift_time": self.retention_time, "drift_reference_time": self.retention_reference,
            "seed": self.seed, "program_epoch": self.program_epoch,
            "output_chunk_tiles": self.output_chunk_tiles, "input_chunk_rows": self.input_chunk_rows,
            "torch_fuse_adc": self.torch_fuse_adc,
            "torch_adc_fusion": self._torch_fusion_status,
            "torch_adc_fusion_error": self._torch_fusion_error,
            "calls": dict(self.execution_counts),
        }
