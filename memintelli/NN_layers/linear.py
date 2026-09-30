# -*- coding:utf-8 -*-
# @File  : linear.py
# @Author: Zhou
# @Date  : 2023/3/22
import os
import sys
import time

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from memintelli.pimpy.data_formats import SlicedData
from memintelli.pimpy.data_formats_multimode import SlicedDataMultiMode
from memintelli.NN_layers.functions import linear_mem_func
from matplotlib import pyplot as plt
from memintelli.pimpy import DPETensor
from memintelli.pimpy.utils import SNR


_STREAM_ATTRS = ('G_indices', 'G', 'max_data', 'e_bias', 'mode1_w_max')


def _input_chunk_max_positions(engine, *, in_features, total_positions):
    input_chunk_size = getattr(engine, "inference_input_chunk_size", None)
    if input_chunk_size is not None and int(input_chunk_size) <= 0:
        return int(total_positions)
    chunk_budget = (
        int(input_chunk_size)
        if input_chunk_size is not None
        else int(getattr(engine, "inference_chunk_size", None) or 32 * 1024 * 1024)
    )
    expansion_factor = 8
    return min(
        int(total_positions),
        max(1, chunk_budget // max(1, int(in_features) * expansion_factor)),
    )


def _make_sliced_data(engine, slice_method, *, device, bw_e, is_weight, paral_size, quant_gran, inference=False):
    mode = getattr(engine, "mode", 0)
    if mode in (1, 2) or getattr(engine, "use_multimode_sliced_data", False):
        return SlicedDataMultiMode(
            slice_method,
            device=device,
            bw_e=bw_e,
            is_weight=is_weight,
            paral_size=paral_size,
            quant_gran=quant_gran,
            inference=inference,
            mode=mode,
        )
    return SlicedData(
        slice_method,
        device=device,
        bw_e=bw_e,
        is_weight=is_weight,
        paral_size=paral_size,
        quant_gran=quant_gran,
        inference=inference,
    )


def _to_cpu_pinned(value):
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(_to_cpu_pinned(item) for item in value)
    if isinstance(value, list):
        return [_to_cpu_pinned(item) for item in value]
    if value.device.type != 'cpu':
        value = value.cpu()
    if not value.is_contiguous() or any(stride == 0 for stride in value.stride()):
        value = value.clone(memory_format=torch.contiguous_format)
    if torch.cuda.is_available() and not value.is_pinned():
        value = value.pin_memory()
    return value


def _to_cpu_buffer(value, *, pin=True):
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(_to_cpu_buffer(item, pin=pin) for item in value)
    if isinstance(value, list):
        return [_to_cpu_buffer(item, pin=pin) for item in value]
    if value.device.type != 'cpu':
        value = value.cpu()
    if not value.is_contiguous() or any(stride == 0 for stride in value.stride()):
        value = value.clone(memory_format=torch.contiguous_format)
    if pin and torch.cuda.is_available() and not value.is_pinned():
        value = value.pin_memory()
    return value


def _pin_cpu_buffer(value):
    return _to_cpu_buffer(value, pin=True)


def _to_device(value, device):
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    return value.to(device, non_blocking=True)


def tensor_bytes(value):
    if value is None:
        return 0
    if isinstance(value, (tuple, list)):
        return sum(tensor_bytes(item) for item in value)
    return value.nelement() * value.element_size()


def tensor_pinned_bytes(value):
    if value is None:
        return 0
    if isinstance(value, (tuple, list)):
        return sum(tensor_pinned_bytes(item) for item in value)
    return tensor_bytes(value) if value.device.type == 'cpu' and value.is_pinned() else 0


class LinearMem(nn.Module):
    _transfer_streams = {}  # Class-level per-device CUDA streams for async prefetch
    _empty_cache_release_counters = {}
    _window_pin_cache_budget_bytes = 0
    _window_pin_cache_used_bytes = 0
    _window_pin_cache_generation = 0
    _execution_trace_active = False
    _execution_trace = []

    @classmethod
    def configure_window_pin_cache(cls, budget_bytes=0):
        cls._window_pin_cache_budget_bytes = max(0, int(budget_bytes or 0))
        cls._window_pin_cache_used_bytes = 0
        cls._window_pin_cache_generation += 1

    @classmethod
    def _get_transfer_stream(cls, device):
        """Get (or create) a dedicated CUDA stream for async data transfers."""
        device = torch.device(device)
        if device.type != "cuda":
            return None
        key = str(device)
        if key not in cls._transfer_streams:
            cls._transfer_streams[key] = torch.cuda.Stream(device=device)
        return cls._transfer_streams[key]

    @classmethod
    def begin_execution_trace(cls):
        cls._execution_trace = []
        cls._execution_trace_active = True

    @classmethod
    def end_execution_trace(cls):
        cls._execution_trace_active = False
        trace = list(getattr(cls, "_execution_trace", []) or [])
        cls._execution_trace = []
        return trace

    def __init__(self, engine, in_features: int, out_features: int, input_slice:[list, tuple], weight_slice:[list, tuple],
                 bias: bool = True, device=None, dtype=torch.float32, bw_e=None, input_paral_size=(1, 32), weight_paral_size=(32, 32),
                 input_quant_gran=(1, 32), weight_quant_gran=(32, 32), skip_initial_mapping=False,
                 triton_output_chunk_limit_override=0):
        '''
        :param in_features: the input neuron number
        :param out_features: the output neuron number
        :param bias: use bias or not, default is True
        :param input_sli_mod: the slice method of the input matrix, default is (1, 1, 2, 4)
        :param weight_sli_mod: the slice method of the weight matrix, default is (1, 1, 2, 4)
        :param bw_e: the bit width of the input and weight, default is None, which means use the INT
        :param device: use cuda or cpu, default is None, which means use cpu
        :param dtype:
        :param skip_initial_mapping: if True, skip the initial weight→conductance mapping.
            Use when weights will be overwritten (e.g. loading pretrained) and update_weight()
            will be called later. Saves significant GPU memory during model initialization.
        '''
        super(LinearMem, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        if not skip_initial_mapping:
            self.reset_parameters()
        self.weight_slice_method = torch.tensor(weight_slice).to(device)
        self.input_slice_method = torch.tensor(input_slice).to(device)

        self.weight_sliced = _make_sliced_data(
            engine,
            self.weight_slice_method,
            device=device,
            bw_e=bw_e,
            is_weight=True,
            paral_size=weight_paral_size,
            quant_gran=weight_quant_gran,
        )
        self.weight_sliced.triton_output_chunk_limit_override = max(
            0, int(triton_output_chunk_limit_override or 0)
        )
        self.engine = engine
        if not skip_initial_mapping:
            self.weight_sliced.slice_data_imp(engine, self.weight.detach().t())
        self.input_paral_size = input_paral_size
        self.input_quant_gran = input_quant_gran
        self.inference_mode = False  # set True via prepare_for_inference()
        self._streaming = False  # CPU offloading mode for streaming inference
        self._input_sliced_cache = None  # reusable SlicedData for inference (avoids re-creation)
        # Use object.__setattr__ to avoid nn.Module registering these as submodules.
        # _next_streaming_layer will hold a reference to another LinearMem (nn.Module),
        # and PyTorch's __setattr__ would auto-register it as a child, creating circular
        # module trees that cause RecursionError in .to() / state_dict() etc.
        object.__setattr__(self, '_next_streaming_layer', None)  # prefetch chain link
        object.__setattr__(self, '_next_restore_prefetch_layer', None)
        object.__setattr__(self, '_prefetch_event', None)  # CUDA event for prefetch sync
        object.__setattr__(self, '_prefetch_start_event', None)
        object.__setattr__(self, '_pinned_buffers', {})  # persistent CPU source buffers
        object.__setattr__(self, '_active_pinned_buffers', {})  # temporary pinned transfer window
        object.__setattr__(self, '_window_cached_buffers', {})  # budgeted reusable pinned staging buffers
        object.__setattr__(self, '_window_pin_cache_enabled', False)
        object.__setattr__(self, '_window_pin_cache_allowed', True)
        object.__setattr__(self, '_window_pin_cache_generation', LinearMem._window_pin_cache_generation)
        object.__setattr__(self, '_streaming_pin_policy', 'persistent')
        object.__setattr__(self, '_empty_cache_after_release', False)
        object.__setattr__(self, '_empty_cache_after_release_interval', 1)
        object.__setattr__(self, '_lazy_prepare', False)
        object.__setattr__(self, '_lazy_prepared', False)
        object.__setattr__(self, '_lazy_streaming', False)
        object.__setattr__(self, '_lazy_free_weights', False)
        object.__setattr__(self, '_lazy_release_after_forward', False)
        object.__setattr__(self, '_runtime_cuda_events', [])
        object.__setattr__(self, '_runtime_counters', {
            'offload_to_cpu_count': 0,
            'offload_to_cpu_bytes': 0,
            'sync_load_count': 0,
            'sync_load_bytes': 0,
            'prefetch_started_count': 0,
            'prefetch_wait_count': 0,
            'prefetch_bytes': 0,
            'prefetch_oom_count': 0,
            'prefetch_complete_on_arrival_count': 0,
            'prefetch_pending_on_arrival_count': 0,
            'next_restore_prefetch_started_count': 0,
            'next_restore_prefetch_skip_count': 0,
            'next_restore_prefetch_skip_streaming_count': 0,
            'next_restore_prefetch_skip_unprepared_count': 0,
            'release_gpu_tensor_count': 0,
            'release_gpu_tensor_bytes': 0,
            'pinned_buffer_peak_bytes': 0,
            'pin_cpu_buffer_ms': 0.0,
            'sync_load_ms': 0.0,
            'prefetch_schedule_ms': 0.0,
            'prefetch_copy_ms': 0.0,
            'prefetch_copy_event_count': 0,
            'release_gpu_tensor_ms': 0.0,
            'window_pin_cache_hit_count': 0,
            'window_pin_cache_miss_count': 0,
            'window_pin_cache_store_count': 0,
            'window_pin_cache_bytes': 0,
            'lazy_prepare_count': 0,
            'input_slice_count': 0,
            'mapreduce_count': 0,
            'postprocess_count': 0,
            'release_prepared_count': 0,
            'activation_slice_fused_success_count': 0,
            'activation_slice_fused_fallback_count': 0,
            'lazy_prepare_ms': 0.0,
            'input_slice_ms': 0.0,
            'mapreduce_ms': 0.0,
            'postprocess_ms': 0.0,
            'release_prepared_ms': 0.0,
            'lazy_prepare_wall_ms': 0.0,
            'input_slice_wall_ms': 0.0,
            'mapreduce_wall_ms': 0.0,
            'postprocess_wall_ms': 0.0,
            'release_prepared_wall_ms': 0.0,
        })

    def _runtime_count(self, key, value=1):
        engine = getattr(self, "engine", None)
        if not bool(getattr(engine, "runtime_counters", True)):
            return
        counters = getattr(self, '_runtime_counters', None)
        if counters is None:
            counters = {}
            object.__setattr__(self, '_runtime_counters', counters)
        counters[key] = counters.get(key, 0) + value

    def _runtime_add_ms(self, key, start):
        self._runtime_count(key, (time.perf_counter() - start) * 1000.0)

    def _runtime_stage_timing_enabled(self):
        engine = getattr(self, "engine", None)
        return bool(getattr(engine, "runtime_stage_timing", False) or getattr(engine, "profile", False))

    def _runtime_stage_start(self):
        if not self._runtime_stage_timing_enabled():
            return None
        device = torch.device(getattr(self.engine, "device", "cpu"))
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                event = torch.cuda.Event(enable_timing=True)
                event.record(torch.cuda.current_stream(device))
                return ("cuda", event, time.perf_counter())
            except Exception:
                pass
        return ("wall", None, time.perf_counter())

    def _runtime_stage_stop(self, token, count_key, ms_key):
        self._runtime_count(count_key)
        if token is None:
            return
        mode, start_event, start_wall = token
        end_wall = time.perf_counter()
        wall_ms = (end_wall - start_wall) * 1000.0
        if ms_key.endswith("_ms"):
            self._runtime_count(ms_key[:-3] + "_wall_ms", wall_ms)
        if mode == "cuda":
            try:
                device = torch.device(getattr(self.engine, "device", "cpu"))
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record(torch.cuda.current_stream(device))
                events = getattr(self, '_runtime_cuda_events', None)
                if events is None:
                    events = []
                    object.__setattr__(self, '_runtime_cuda_events', events)
                events.append((ms_key, start_event, end_event, wall_ms))
                return
            except Exception:
                pass
        self._runtime_count(ms_key, wall_ms)

    def _runtime_finalize_events(self):
        events = getattr(self, '_runtime_cuda_events', None) or []
        if not events:
            return
        for ms_key, start_event, end_event, fallback_ms in events:
            try:
                end_event.synchronize()
                elapsed_ms = float(start_event.elapsed_time(end_event))
            except Exception:
                elapsed_ms = float(fallback_ms or 0.0)
            self._runtime_count(ms_key, elapsed_ms)
        events.clear()

    def reset_runtime_mechanism_counters(self):
        counters = getattr(self, '_runtime_counters', None)
        if counters is None:
            counters = {}
            object.__setattr__(self, '_runtime_counters', counters)
        for key, value in list(counters.items()):
            counters[key] = 0.0 if isinstance(value, float) else 0
        object.__setattr__(self, '_runtime_cuda_events', [])
        return self

    def _record_prefetch_elapsed_if_ready(self):
        start_event = getattr(self, '_prefetch_start_event', None)
        end_event = getattr(self, '_prefetch_event', None)
        if start_event is None or end_event is None:
            return
        try:
            if end_event.query():
                self._runtime_count('prefetch_copy_ms', float(start_event.elapsed_time(end_event)))
                self._runtime_count('prefetch_copy_event_count')
                object.__setattr__(self, '_prefetch_start_event', None)
        except Exception:
            pass

    def _runtime_pinned_bytes(self):
        pinned = sum(tensor_pinned_bytes(value) for value in getattr(self, '_pinned_buffers', {}).values())
        pinned += sum(tensor_pinned_bytes(value) for value in getattr(self, '_active_pinned_buffers', {}).values())
        pinned += sum(tensor_pinned_bytes(value) for value in getattr(self, '_window_cached_buffers', {}).values())
        return pinned

    def _runtime_update_pinned_peak(self):
        counters = getattr(self, '_runtime_counters', None)
        if counters is None:
            counters = {}
            object.__setattr__(self, '_runtime_counters', counters)
        counters['pinned_buffer_peak_bytes'] = max(
            counters.get('pinned_buffer_peak_bytes', 0),
            self._runtime_pinned_bytes(),
        )

    def runtime_mechanism_counters(self):
        self._runtime_finalize_events()
        counters = dict(getattr(self, '_runtime_counters', {}) or {})
        current_pinned = self._runtime_pinned_bytes()
        counters['current_pinned_buffer_bytes'] = current_pinned
        counters['pinned_buffer_peak_bytes'] = max(counters.get('pinned_buffer_peak_bytes', 0), current_pinned)
        counters['pinned_buffer_count'] = (
            len(getattr(self, '_pinned_buffers', {}) or {})
            + len(getattr(self, '_active_pinned_buffers', {}) or {})
            + len(getattr(self, '_window_cached_buffers', {}) or {})
        )
        counters['has_prefetch_link'] = self._next_streaming_layer is not None
        counters['has_next_restore_prefetch_link'] = self._next_restore_prefetch_layer is not None
        counters['has_pending_prefetch_event'] = self._prefetch_event is not None
        counters['lazy_prepared'] = bool(self._lazy_prepared)
        counters['streaming'] = bool(self._streaming)
        counters['streaming_pin_policy'] = getattr(self, '_streaming_pin_policy', 'persistent')
        counters['window_pin_cache_enabled'] = bool(getattr(self, '_window_pin_cache_enabled', False))
        counters['window_pin_cache_allowed'] = bool(getattr(self, '_window_pin_cache_allowed', True))
        return counters

    def reset_parameters(self) -> None:
        # Setting x=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if bool(getattr(type(self), "_execution_trace_active", False)):
            type(self)._execution_trace.append(self)
        token = (
            self.engine._profile_start("linearmem_forward_total")
            if hasattr(self.engine, "_profile_start")
            else None
        )
        try:
            if self.inference_mode:
                return self._forward_inference(input)
            input_sliced = _make_sliced_data(
                self.engine,
                self.input_slice_method,
                device=input.device,
                bw_e=self.weight_sliced.bw_e,
                is_weight=False,
                paral_size=self.input_paral_size,
                quant_gran=self.input_quant_gran,
            )
            input_sliced.slice_data_imp(self.engine, input.detach())
            return linear_mem_func(self.engine, input, self.weight, input_sliced, self.weight_sliced, self.bias)
        finally:
            if token is not None and hasattr(self.engine, "_profile_stop"):
                self.engine._profile_stop(token, inference_mode=bool(self.inference_mode))

    def _forward_inference(self, input: torch.Tensor) -> torch.Tensor:
        """Optimized inference forward with async prefetch support.

        Execution flow for streaming layers:
          1. If previous layer started a prefetch → just wait for the event (near-zero cost)
             Otherwise (first layer) → synchronous load from pinned CPU
          2. Compute MapReduceDot
          3. Start async prefetch for next streaming layer (overlaps with norms/attention)
          4. Release GPU tensors (point references back to pinned buffers — NO D2H copy!)
        """
        if getattr(self.engine, "require_bf16_input", False) and input.dtype != torch.bfloat16:
            raise ValueError("This execution profile requires BF16 input activations")
        if self._lazy_prepare and not self._lazy_prepared:
            token_prepare = self._runtime_stage_start()
            try:
                self._prepare_inference_weight(
                    streaming=self._lazy_streaming,
                    free_weights=self._lazy_free_weights,
                    pin_policy=self._streaming_pin_policy,
                )
            finally:
                self._runtime_stage_stop(token_prepare, 'lazy_prepare_count', 'lazy_prepare_ms')

        # Streaming: ensure G data is on GPU
        if self._streaming:
            if self._prefetch_event is not None:
                self._record_prefetch_elapsed_if_ready()
                try:
                    arrived_ready = bool(self._prefetch_event.query())
                except Exception:
                    arrived_ready = False
                self._runtime_count(
                    'prefetch_complete_on_arrival_count' if arrived_ready else 'prefetch_pending_on_arrival_count'
                )
                # Async prefetch was started by the previous layer — wait on GPU stream only
                self._runtime_count('prefetch_wait_count')
                torch.cuda.current_stream(self.engine.device).wait_event(self._prefetch_event)
                self._prefetch_event = None
            else:
                # First streaming layer in this forward pass — synchronous load
                self._load_to_device(self.engine.device)

        input_shape = input.shape
        in_features = max(1, input_shape[-1])
        input_2d = input.reshape(-1, in_features)

        # Reuse cached SlicedData object (avoids creating new one each call)
        if self._input_sliced_cache is None:
            self._input_sliced_cache = _make_sliced_data(
                self.engine,
                self.input_slice_method,
                device=input_2d.device,
                bw_e=self.weight_sliced.bw_e, is_weight=False,
                paral_size=self.input_paral_size,
                quant_gran=self.input_quant_gran,
                inference=True)
            if bool(getattr(self.engine, 'triton_fuse_activation_slices', False)):
                setattr(self._input_sliced_cache, 'enable_triton_activation_slicing', True)
        input_sliced = self._input_sliced_cache
        # Mode-0 2-D Linear kernels operate on the flattened token matrix.
        # Keep the SlicedData logical shape aligned with input_2d so direct-final
        # paths can consume the standard [N, M, I, J, K] tiled layout.
        if getattr(self.engine, "mode", 0) == 0:
            input_sliced.shape = input_2d.shape

        # Input-dimension chunking to bound peak memory during slice_data_imp().
        # This complements engine-side matmul chunking (which chunks along weight cols).
        max_positions = _input_chunk_max_positions(
            self.engine,
            in_features=in_features,
            total_positions=input_2d.shape[0],
        )
        if getattr(self.engine, "mode", 0) == 1:
            # Mode 1 uses a per-forward signed-DAC activation scale. Splitting
            # rows would change that scale and therefore change the simulated ADC.
            max_positions = input_2d.shape[0]
        total_positions = input_2d.shape[0]

        if max_positions >= total_positions:
            if hasattr(self.engine, "schedule_mode0_restore_input_prefetch"):
                self.engine.schedule_mode0_restore_input_prefetch(input_2d, self.weight_sliced)
            self._schedule_next_restore_prefetch(input_2d)
            token_slice = self._runtime_stage_start()
            try:
                if hasattr(self.engine, "probe_activation_slice_reuse"):
                    self.engine.probe_activation_slice_reuse(
                        input_2d,
                        self.input_slice_method,
                        self.input_paral_size,
                        self.input_quant_gran,
                        getattr(self.engine, "mode", 0),
                    )
                cache_entry = (
                    self.engine.lookup_activation_slice_cache(
                        input_2d,
                        self.input_slice_method,
                        self.input_paral_size,
                        self.input_quant_gran,
                        getattr(self.engine, "mode", 0),
                    )
                    if hasattr(self.engine, "lookup_activation_slice_cache")
                    else None
                )
                if cache_entry is not None:
                    input_sliced.sliced_data = cache_entry["sliced_data"]
                    input_sliced.max_data = cache_entry["max_data"]
                    input_sliced.precomputed_v_sliced = cache_entry.get("precomputed_v_sliced")
                    input_sliced.e_bias = cache_entry.get("e_bias")
                    input_sliced.quantized_data = None
                    input_sliced.shape = input_2d.shape
                    input_sliced.activation_slice_fused = bool(cache_entry.get("activation_slice_fused", False))
                else:
                    input_sliced.slice_data_imp(self.engine, input_2d)
                    if hasattr(self.engine, "store_activation_slice_cache"):
                        self.engine.store_activation_slice_cache(
                            input_2d,
                            input_sliced,
                            self.input_slice_method,
                            self.input_paral_size,
                            self.input_quant_gran,
                            getattr(self.engine, "mode", 0),
                        )
            finally:
                self._runtime_stage_stop(token_slice, 'input_slice_count', 'input_slice_ms')
                self._record_activation_slice_path(input_sliced)
            token_mapreduce = self._runtime_stage_start()
            try:
                output = self.engine.MapReduceDot(input_sliced, self.weight_sliced)
            finally:
                self._runtime_stage_stop(token_mapreduce, 'mapreduce_count', 'mapreduce_ms')
            # Free input data immediately (keep the cache object structure)
            input_sliced.sliced_data = None
            input_sliced.max_data = None
            input_sliced.precomputed_v_sliced = None
            input_sliced.e_bias = None
        else:
            out_chunks = []
            for start in range(0, total_positions, max_positions):
                end = min(start + max_positions, total_positions)
                x_chunk = input_2d[start:end, :]
                token_slice = self._runtime_stage_start()
                try:
                    if hasattr(self.engine, "probe_activation_slice_reuse"):
                        self.engine.probe_activation_slice_reuse(
                            x_chunk,
                            self.input_slice_method,
                            self.input_paral_size,
                            self.input_quant_gran,
                            getattr(self.engine, "mode", 0),
                        )
                    cache_entry = (
                        self.engine.lookup_activation_slice_cache(
                            x_chunk,
                            self.input_slice_method,
                            self.input_paral_size,
                            self.input_quant_gran,
                            getattr(self.engine, "mode", 0),
                        )
                        if hasattr(self.engine, "lookup_activation_slice_cache")
                        else None
                    )
                    if cache_entry is not None:
                        input_sliced.sliced_data = cache_entry["sliced_data"]
                        input_sliced.max_data = cache_entry["max_data"]
                        input_sliced.precomputed_v_sliced = cache_entry.get("precomputed_v_sliced")
                        input_sliced.e_bias = cache_entry.get("e_bias")
                        input_sliced.quantized_data = None
                        input_sliced.shape = x_chunk.shape
                        input_sliced.activation_slice_fused = bool(cache_entry.get("activation_slice_fused", False))
                    else:
                        input_sliced.slice_data_imp(self.engine, x_chunk)
                        if hasattr(self.engine, "store_activation_slice_cache"):
                            self.engine.store_activation_slice_cache(
                                x_chunk,
                                input_sliced,
                                self.input_slice_method,
                                self.input_paral_size,
                                self.input_quant_gran,
                                getattr(self.engine, "mode", 0),
                            )
                finally:
                    self._runtime_stage_stop(token_slice, 'input_slice_count', 'input_slice_ms')
                    self._record_activation_slice_path(input_sliced)
                token_mapreduce = self._runtime_stage_start()
                try:
                    out_chunks.append(self.engine.MapReduceDot(input_sliced, self.weight_sliced))
                finally:
                    self._runtime_stage_stop(token_mapreduce, 'mapreduce_count', 'mapreduce_ms')
                input_sliced.sliced_data = None
                input_sliced.max_data = None
                input_sliced.precomputed_v_sliced = None
                input_sliced.e_bias = None
            output = torch.cat(out_chunks, dim=0)
            del out_chunks

        token_postprocess = self._runtime_stage_start()
        output = output.reshape(*input_shape[:-1], self.out_features)

        # Async prefetch next streaming layer (overlaps with attention/norms between layers)
        if self._next_streaming_layer is not None and self._next_streaming_layer._streaming:
            self._next_streaming_layer._async_prefetch(self.engine.device)

        # Release GPU tensors — point references back to pinned CPU buffers (NO D2H copy!)
        if self._streaming:
            self._release_gpu_tensors()

        # Move output back to input device if engine is on a different device
        if output.device != input.device:
            output = output.to(input.device)

        output_dtype_policy = getattr(self.engine, "linear_output_dtype", "input")
        if output_dtype_policy in ("auto", "input", "keep"):
            output_dtype = input.dtype
        else:
            output_dtype = output_dtype_policy
        if output.dtype != output_dtype:
            output = output.to(output_dtype)

        if self.bias is not None:
            output = output + self.bias
        self._runtime_stage_stop(token_postprocess, 'postprocess_count', 'postprocess_ms')
        if self._lazy_release_after_forward:
            token_release = self._runtime_stage_start()
            try:
                self.release_prepared_weight()
            finally:
                self._runtime_stage_stop(token_release, 'release_prepared_count', 'release_prepared_ms')
        return output

    def _schedule_next_restore_prefetch(self, input_2d):
        engine = getattr(self, "engine", None)
        if not bool(getattr(engine, "triton_cross_linear_restore_prefetch", False)):
            return
        next_layer = getattr(self, "_next_restore_prefetch_layer", None)
        if next_layer is None or next_layer is self:
            self._runtime_count('next_restore_prefetch_skip_count')
            return
        if bool(getattr(next_layer, "_streaming", False)):
            self._runtime_count('next_restore_prefetch_skip_streaming_count')
            return
        if bool(getattr(next_layer, "_lazy_prepare", False)) and not bool(getattr(next_layer, "_lazy_prepared", False)):
            self._runtime_count('next_restore_prefetch_skip_unprepared_count')
            return
        if not bool(getattr(next_layer, "inference_mode", False)):
            self._runtime_count('next_restore_prefetch_skip_unprepared_count')
            return
        if getattr(next_layer, "engine", None) is not engine:
            self._runtime_count('next_restore_prefetch_skip_count')
            return
        mat = getattr(next_layer, "weight_sliced", None)
        if mat is None or not hasattr(engine, "schedule_mode0_restore_input_prefetch"):
            self._runtime_count('next_restore_prefetch_skip_count')
            return
        if engine.schedule_mode0_restore_input_prefetch(input_2d, mat, prefetch_source="next_linear"):
            self._runtime_count('next_restore_prefetch_started_count')
        else:
            self._runtime_count('next_restore_prefetch_skip_count')

    def _record_activation_slice_path(self, input_sliced):
        if bool(getattr(self.engine, 'triton_fuse_activation_slices', False)):
            key = (
                'activation_slice_fused_success_count'
                if bool(getattr(input_sliced, 'activation_slice_fused', False))
                else 'activation_slice_fused_fallback_count'
            )
            self._runtime_count(key)

    def enable_lazy_inference(self, streaming=False, free_weights=False, release_after_forward=True, pin_policy='persistent'):
        """Defer weight conductance mapping until the layer is first executed."""
        if free_weights and release_after_forward:
            raise ValueError("free_weights=True is incompatible with lazy release_after_forward=True.")
        self.inference_mode = True
        self.weight_sliced.inference = True
        object.__setattr__(self, '_lazy_prepare', True)
        object.__setattr__(self, '_lazy_prepared', False)
        object.__setattr__(self, '_lazy_streaming', bool(streaming))
        object.__setattr__(self, '_lazy_free_weights', bool(free_weights))
        object.__setattr__(self, '_lazy_release_after_forward', bool(release_after_forward))
        object.__setattr__(self, '_streaming_pin_policy', str(pin_policy))
        object.__setattr__(self, '_next_streaming_layer', None)
        object.__setattr__(self, '_next_restore_prefetch_layer', None)
        return self

    def _prepare_inference_weight(self, streaming=False, free_weights=False, pin_policy=None):
        """Map the current weight to conductance for prepared inference."""
        self.inference_mode = True
        self.weight_sliced.inference = True
        if self.bias is not None and self.bias.device != self.engine.device:
            self.bias.data = self.bias.data.to(self.engine.device)
        if self.input_slice_method.device != self.engine.device:
            self.input_slice_method = self.input_slice_method.to(self.engine.device)
        if self.weight_slice_method.device != self.engine.device:
            self.weight_slice_method = self.weight_slice_method.to(self.engine.device)

        self.update_weight()
        if getattr(self.engine, "write_variation", 0) == 0 and hasattr(self.weight_sliced, "compress_G"):
            self.weight_sliced.compress_G(self.engine)
        if hasattr(self.weight_sliced, "quantized_data"):
            self.weight_sliced.quantized_data = None
        if hasattr(self.weight_sliced, "sliced_data"):
            self.weight_sliced.sliced_data = None

        if free_weights:
            self.weight.data = torch.empty(0, device="cpu", dtype=self.weight.dtype)
        direct_release = bool(streaming and getattr(self, '_lazy_release_after_forward', False))
        if streaming and not direct_release:
            self._offload_to_cpu(pin_policy=pin_policy)
            self._streaming = True
        else:
            self._streaming = False
        object.__setattr__(self, '_lazy_prepared', True)
        return self

    def release_prepared_weight(self):
        """Drop mapped conductance buffers after a lazy inference call."""
        self._release_gpu_tensors()
        ws = self.weight_sliced
        for attr in _STREAM_ATTRS:
            setattr(ws, attr, None)
        self._pinned_buffers.clear()
        self._active_pinned_buffers.clear()
        self._clear_window_pin_cache()
        object.__setattr__(self, '_prefetch_event', None)
        object.__setattr__(self, '_prefetch_start_event', None)
        object.__setattr__(self, '_streaming', False)
        object.__setattr__(self, '_lazy_prepared', False)
        self._empty_cache_after_release_if_due()

    def _offload_to_cpu(self, pin_policy=None):
        """Initial offload: move G data from GPU to pinned CPU memory.

        Called ONCE during update_weight_and_prepare(streaming=True).
        In persistent mode, this creates pinned CPU buffers for fast async
        prefetch. In window mode, it keeps pageable CPU source buffers and
        pins only the current/prefetched layer on demand.
        """
        ws = self.weight_sliced
        device = torch.device(getattr(self.engine, "device", "cpu"))
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)  # ensure G computation is complete
        if pin_policy is not None:
            object.__setattr__(self, '_streaming_pin_policy', str(pin_policy))
        use_persistent_pin = getattr(self, '_streaming_pin_policy', 'persistent') == 'persistent'
        self._active_pinned_buffers.clear()
        self._clear_window_pin_cache()
        object.__setattr__(
            self,
            '_window_pin_cache_enabled',
            (not use_persistent_pin)
            and LinearMem._window_pin_cache_budget_bytes > 0
            and bool(getattr(self, '_window_pin_cache_allowed', True)),
        )

        for attr in _STREAM_ATTRS:
            tensor = getattr(ws, attr, None)
            if tensor is not None:
                cpu_t = _to_cpu_buffer(tensor, pin=use_persistent_pin)
                self._pinned_buffers[attr] = cpu_t
                setattr(ws, attr, cpu_t)
        bytes_moved = sum(tensor_bytes(value) for value in self._pinned_buffers.values())
        if bytes_moved:
            self._runtime_count('offload_to_cpu_count')
            self._runtime_count('offload_to_cpu_bytes', bytes_moved)
            self._runtime_update_pinned_peak()

    def _clear_window_pin_cache(self):
        cached = getattr(self, '_window_cached_buffers', {}) or {}
        bytes_cached = sum(tensor_pinned_bytes(value) for value in cached.values())
        if bytes_cached:
            LinearMem._window_pin_cache_used_bytes = max(
                0,
                LinearMem._window_pin_cache_used_bytes - bytes_cached,
            )
        cached.clear()

    def _sync_window_pin_cache_generation(self):
        generation = LinearMem._window_pin_cache_generation
        if getattr(self, '_window_pin_cache_generation', None) != generation:
            self._clear_window_pin_cache()
            object.__setattr__(self, '_window_pin_cache_generation', generation)

    def _maybe_store_window_pin_cache(self, attr, pinned):
        if not bool(getattr(self, '_window_pin_cache_enabled', False)):
            return False
        if not bool(getattr(self, '_window_pin_cache_allowed', True)):
            return False
        self._sync_window_pin_cache_generation()
        size = tensor_pinned_bytes(pinned) or tensor_bytes(pinned)
        if size <= 0:
            return False
        if LinearMem._window_pin_cache_used_bytes + size > LinearMem._window_pin_cache_budget_bytes:
            return False
        cached = getattr(self, '_window_cached_buffers', None)
        if cached is None:
            cached = {}
            object.__setattr__(self, '_window_cached_buffers', cached)
        cached[attr] = pinned
        LinearMem._window_pin_cache_used_bytes += size
        self._runtime_count('window_pin_cache_store_count')
        self._runtime_count('window_pin_cache_bytes', size)
        self._runtime_update_pinned_peak()
        return True

    def _transfer_source_buffer(self, attr):
        source = self._pinned_buffers.get(attr)
        if source is None:
            return None
        if getattr(self, '_streaming_pin_policy', 'persistent') == 'persistent':
            return source
        if bool(getattr(self, '_window_pin_cache_enabled', False)):
            self._sync_window_pin_cache_generation()
            cached = getattr(self, '_window_cached_buffers', {}) or {}
            pinned = cached.get(attr)
            if pinned is not None:
                self._runtime_count('window_pin_cache_hit_count')
                return pinned
        start = time.perf_counter()
        pinned = _pin_cpu_buffer(source)
        self._runtime_add_ms('pin_cpu_buffer_ms', start)
        self._runtime_count('window_pin_cache_miss_count')
        if not self._maybe_store_window_pin_cache(attr, pinned):
            self._active_pinned_buffers[attr] = pinned
        return pinned

    def _drop_active_pinned_buffers(self):
        self._active_pinned_buffers.clear()
        self._runtime_update_pinned_peak()

    def _load_to_device(self, device):
        """Synchronous load: copy pinned CPU → GPU, blocks until complete.

        Used for the first streaming layer in a forward pass (no prefetch available).
        Subsequent layers should use _async_prefetch for overlap.
        """
        start = time.perf_counter()
        ws = self.weight_sliced
        device = torch.device(device)
        bytes_moved = 0
        for attr in _STREAM_ATTRS:
            pinned = self._transfer_source_buffer(attr)
            if pinned is not None:
                bytes_moved += tensor_bytes(pinned)
                setattr(ws, attr, _to_device(pinned, device))
        if bytes_moved:
            self._runtime_update_pinned_peak()
            self._runtime_count('sync_load_count')
            self._runtime_count('sync_load_bytes', bytes_moved)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        if bytes_moved:
            self._runtime_add_ms('sync_load_ms', start)

    def _async_prefetch(self, device):
        """Start loading G data to GPU on a dedicated transfer stream (non-blocking).

        This is called by the PREVIOUS layer in the prefetch chain, so the H2D
        transfer overlaps with whatever computation happens between the two layers
        (attention, norms, residuals, etc.).

        OOM-safe: if GPU doesn't have enough free memory for the prefetch allocation,
        we reset back to pinned CPU buffers and clear the event. The next layer's
        forward() will then fall back to synchronous loading (which works because
        the current layer's GPU tensors will have been released by then).
        """
        try:
            start = time.perf_counter()
            device = torch.device(device)
            if device.type != "cuda":
                self._load_to_device(device)
                object.__setattr__(self, '_prefetch_event', None)
                object.__setattr__(self, '_prefetch_start_event', None)
                return
            stream = LinearMem._get_transfer_stream(device)
            ws = self.weight_sliced
            bytes_moved = 0
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                start_event.record(stream)
                for attr in _STREAM_ATTRS:
                    pinned = self._transfer_source_buffer(attr)
                    if pinned is not None:
                        bytes_moved += tensor_bytes(pinned)
                        setattr(ws, attr, _to_device(pinned, device))
                end_event.record(stream)
            if bytes_moved:
                self._runtime_update_pinned_peak()
                self._runtime_count('prefetch_started_count')
                self._runtime_count('prefetch_bytes', bytes_moved)
                object.__setattr__(self, '_prefetch_start_event', start_event)
                object.__setattr__(self, '_prefetch_event', end_event)
                self._runtime_add_ms('prefetch_schedule_ms', start)
            else:
                object.__setattr__(self, '_prefetch_start_event', None)
                object.__setattr__(self, '_prefetch_event', None)
        except torch.cuda.OutOfMemoryError:
            # Prefetch failed — reset to pinned CPU, next forward will sync-load
            self._runtime_count('prefetch_oom_count')
            self._release_gpu_tensors()
            object.__setattr__(self, '_prefetch_start_event', None)
            object.__setattr__(self, '_prefetch_event', None)
            torch.cuda.empty_cache()

    def _release_gpu_tensors(self):
        """Release GPU tensors by pointing references back to CPU source buffers.

        KEY INSIGHT: G data never changes during inference, so we DON'T need
        D2H copies. We just set the SlicedData attributes back to the persistent
        pinned CPU tensors. The old GPU tensors become unreferenced and are freed
        by PyTorch's caching allocator.

        This eliminates 50% of the PCIe transfer overhead compared to the old
        approach (which did GPU→CPU copy + re-pin on every forward pass).
        """
        start = time.perf_counter()
        self._record_prefetch_elapsed_if_ready()
        ws = self.weight_sliced
        bytes_released = 0
        for attr in _STREAM_ATTRS:
            source = self._pinned_buffers.get(attr)
            if source is not None:
                bytes_released += tensor_bytes(source)
                setattr(ws, attr, source)
        if getattr(self, '_streaming_pin_policy', 'persistent') != 'persistent':
            self._drop_active_pinned_buffers()
        if bytes_released:
            self._runtime_count('release_gpu_tensor_count')
            self._runtime_count('release_gpu_tensor_bytes', bytes_released)
            self._runtime_add_ms('release_gpu_tensor_ms', start)
            self._empty_cache_after_release_if_due()

    def _empty_cache_after_release_if_due(self):
        if not bool(getattr(self, '_empty_cache_after_release', False)):
            return
        device = torch.device(getattr(self.engine, "device", "cpu"))
        if device.type != "cuda" or not torch.cuda.is_available():
            return
        interval = max(1, int(getattr(self, '_empty_cache_after_release_interval', 1) or 1))
        key = str(device)
        count = LinearMem._empty_cache_release_counters.get(key, 0) + 1
        LinearMem._empty_cache_release_counters[key] = count
        if count % interval == 0:
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    def update_weight(self):
        self.weight_sliced.slice_data_imp(self.engine, self.weight.detach().t().to(self.engine.device))

def _test(mode=0):
    if mode == 0:
        print("-----------------cuda-----------------")
        engine = DPETensor(
        HGS=1e-5,                       # High conductance state
        LGS=1e-8,                       # Low conductance state
        write_variation=0.0,          # Write variation
        rate_stuck_HGS=0.001,          # Rate of stuck at HGS
        rate_stuck_LGS=0.000,          # Rate of stuck at LGS
        read_variation={0:0.05, 1:0.05, 2:0.05, 3:0.05},           # Read variation
        vnoise=0.05,                   # Random Gaussian noise of voltage
        rdac=2**2,                      # Number of DAC resolution
        g_level=2**2,                   # Number of conductance levels
        radc=2**12
        )
        xblk = [1, 1, 2, 2]
        mblk = [1, 1, 2, 2]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input = torch.randn(500, 100, requires_grad=True).to(device)
        layer = LinearMem(engine, 100, 300, bias=False, input_slice=xblk, weight_slice=mblk, device=device, bw_e=None, input_paral_size=(1, 32), weight_paral_size=(32, 32),
                 input_quant_gran=(1, 32), weight_quant_gran=(32, 32))
        output = layer(input)
        #output.backward(torch.ones_like(output, dtype=torch.float))
        weight = layer.weight.data
        #weight.requires_grad = True
        output_ideal = F.linear(input.to(device), weight)
        output_ideal.backward(torch.ones_like(output_ideal, dtype=torch.float))

        output = output.cpu().detach().numpy()
        output_ideal = output_ideal.cpu().detach().numpy()
        print(SNR(output_ideal, output))


if __name__ == '__main__':
    _test(0)
