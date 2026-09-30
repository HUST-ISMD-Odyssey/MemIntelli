# -*- coding:utf-8 -*-
# @File  : convolution.py
# @Author: Zhou
# @Date  : 2023/11/27

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
import time
import sys,os

from memintelli.pimpy.data_formats import SlicedData
from memintelli.pimpy.data_formats_multimode import SlicedDataMultiMode
from memintelli.NN_layers.functions import conv1d_mem_func, conv2d_mem_func
from memintelli.pimpy import DPETensor


_STREAM_ATTRS = ('G_indices', 'G', 'max_data', 'e_bias', 'mode1_w_max')


def _make_sliced_data(engine, slice_method, *, device, bw_e, is_weight, paral_size, quant_gran, inference=False):
    mode = getattr(engine, "mode", 0)
    if mode in (1, 2):
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
    if value.device.type != 'cpu':
        value = value.cpu()
    if torch.cuda.is_available() and not value.is_pinned():
        value = value.pin_memory()
    return value


def _to_device(value, device):
    if value is None:
        return None
    if isinstance(value, tuple):
        return tuple(_to_device(item, device) for item in value)
    return value.to(device, non_blocking=True)

class Conv1dMem(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,  input_slice:[list, tuple], weight_slice:[list, tuple],
                 stride=1, padding=0, dilation=1, groups=1, bias=True, device=None, dtype=None):
        super(Conv1dMem, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Conv1dMem, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, kernel_size), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting x=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, dot_engine, input: torch.Tensor) -> torch.Tensor:
        # self.weight_sliced.quantize_data_imp(dot_engine, self.weight_sliced.data)
        self.weight_sliced.slice_data_imp(dot_engine, self.weight_sliced.data)
        return conv1d_mem_func(dot_engine, input, self.weight, self.bias, self.stride)


class Conv2dMem(nn.Module):
    _transfer_stream = None  # Class-level CUDA stream for async prefetch

    @classmethod
    def _get_transfer_stream(cls, device):
        """Get (or create) a dedicated CUDA stream for async data transfers."""
        if cls._transfer_stream is None:
            cls._transfer_stream = torch.cuda.Stream(device=device)
        return cls._transfer_stream

    def __init__(self, engine, in_channels, out_channels, kernel_size, input_slice:[list, tuple, torch.Tensor],
                 weight_slice:[list, tuple], stride=1, padding=0, dilation=1,bias=True, device=None, dtype=None,
                  bw_e=None, input_paral_size=(1, 32), weight_paral_size=(32, 32),
                  input_quant_gran=(1, 32), weight_quant_gran=(32, 32), skip_initial_mapping=False):
        super(Conv2dMem, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = nn.Parameter(torch.empty((out_channels, in_channels, kernel_size, kernel_size), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        self.weight_slice_method = torch.tensor(weight_slice).to(device)
        self.input_slice_method = torch.tensor(input_slice).to(device)

        self.input_paral_size = input_paral_size
        self.input_quant_gran = input_quant_gran
        self.weight_sliced = _make_sliced_data(
            engine,
            self.weight_slice_method,
            device=device,
            bw_e=bw_e,
            is_weight=True,
            paral_size=weight_paral_size,
            quant_gran=weight_quant_gran,
        )
        self.engine = engine
        if not skip_initial_mapping:
            # the sliced weight shape is (C_in*kh*kw, C_out)
            self.weight_sliced.slice_data_imp(engine, self.weight.reshape(self.weight.shape[0], -1).detach().t())
        self.inference_mode = False  # set True via prepare_for_inference()
        self._streaming = False  # CPU offloading mode for streaming inference
        self._input_sliced_cache = None  # reusable SlicedData for inference
        # Store kernel dims for use after weight is freed
        self._kernel_h = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
        self._kernel_w = kernel_size if isinstance(kernel_size, int) else kernel_size[1]
        # Async prefetch support (same pattern as LinearMem)
        object.__setattr__(self, '_next_streaming_layer', None)  # prefetch chain link
        object.__setattr__(self, '_prefetch_event', None)  # CUDA event for prefetch sync
        object.__setattr__(self, '_pinned_buffers', {})  # persistent pinned CPU buffers

    def reset_parameters(self) -> None:
        # Setting x=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.inference_mode:
            return self._forward_inference(input)
        # input_unfold size: (N, C*kh*kw, L), N is the batch size, C is the channel, kh and kw is the kernel size
        # L is the length of the unfolded vector, L = H_out * W_out
        # transpose the input_unfold to (N, L, C*kh*kw)
        input_sliced = _make_sliced_data(
            self.engine,
            self.input_slice_method,
            device=input.device,
            bw_e=self.weight_sliced.bw_e,
            is_weight=False,
            paral_size=self.input_paral_size,
            quant_gran=self.input_quant_gran,
        )
        input_unfold = F.unfold(input, kernel_size=self.weight.shape[2:], stride=self.stride, padding=self.padding,
                                dilation=self.dilation).transpose(1, 2)
        input_sliced.slice_data_imp(self.engine, input_unfold.detach())
        return conv2d_mem_func(self.engine, input, self.weight, input_sliced, self.weight_sliced, self.bias,
                               self.stride, self.padding, self.dilation)

    def _forward_inference(self, input: torch.Tensor) -> torch.Tensor:
        """Optimized inference forward with async prefetch + spatial chunking.

        Execution flow for streaming layers (same pattern as LinearMem):
          1. If previous layer started a prefetch → just wait for the event (near-zero cost)
             Otherwise (first layer) → synchronous load from pinned CPU
          2. Unfold input → optionally chunk spatial dimension to avoid OOM
          3. For each spatial chunk: slice_data_imp → MapReduceDot → collect
          4. Start async prefetch for next streaming layer (overlaps with BN/ReLU/residual)
          5. Release GPU tensors (point references back to pinned buffers — NO D2H copy!)

        Spatial chunking is critical for VGG-style networks where early conv layers
        have large spatial dims (224x224) with many channels (64-512), causing
        the unfolded input to exceed GPU memory during quantization/slicing.
        """
        # Streaming: ensure G data is on GPU
        if self._streaming:
            if self._prefetch_event is not None:
                # Async prefetch was started by the previous layer — wait on GPU stream only
                torch.cuda.current_stream(self.engine.device).wait_event(self._prefetch_event)
                self._prefetch_event = None
            else:
                # First streaming layer in this forward pass — synchronous load
                self._load_to_device(self.engine.device)

        # Use saved kernel dims (safe even after weight is freed)
        kernel_size = (self._kernel_h, self._kernel_w)

        # Reuse cached SlicedData object
        if self._input_sliced_cache is None:
            self._input_sliced_cache = _make_sliced_data(
                self.engine,
                self.input_slice_method,
                device=input.device,
                bw_e=self.weight_sliced.bw_e, is_weight=False,
                paral_size=self.input_paral_size,
                quant_gran=self.input_quant_gran,
                inference=True)
        input_sliced = self._input_sliced_cache

        input_unfold = F.unfold(input, kernel_size=kernel_size,
                                stride=self.stride, padding=self.padding,
                                dilation=self.dilation).transpose(1, 2)
        # input_unfold shape: (B, L, C_in*kh*kw) where L = H_out * W_out

        B, L, C_dim = input_unfold.shape

        # ─── Spatial chunking to prevent OOM during input slicing ───
        # For VGG-like nets, early layers have L=50176 (224x224) with C_dim=576 (64*3*3).
        # slice_data_imp expands this by num_slices (~4x) and quantization granularity,
        # creating intermediate tensors of 6-8 GB. Chunking the spatial dimension
        # keeps peak memory bounded.
        chunk_budget = getattr(self.engine, 'inference_chunk_size', None) or 32 * 1024 * 1024
        # Conservative: slicing can expand by ~8x (4 slices * 2x for intermediates)
        elements_per_pos = B * C_dim
        expansion_factor = 8  # conservative estimate for slicing expansion
        max_positions = max(1, chunk_budget // max(1, elements_per_pos * expansion_factor))
        max_positions = min(max_positions, L)

        if max_positions >= L:
            # Small enough to process all spatial positions at once
            input_sliced.slice_data_imp(self.engine, input_unfold)
            del input_unfold
            out = self.engine.MapReduceDot(input_sliced, self.weight_sliced)
            input_sliced.sliced_data = None
            input_sliced.max_data = None
            input_sliced.e_bias = None
        else:
            # Process spatial positions in chunks to avoid OOM
            out_chunks = []
            for start in range(0, L, max_positions):
                end = min(start + max_positions, L)
                chunk = input_unfold[:, start:end, :]
                input_sliced.slice_data_imp(self.engine, chunk)
                out_chunk = self.engine.MapReduceDot(input_sliced, self.weight_sliced)
                out_chunks.append(out_chunk)
                input_sliced.sliced_data = None
                input_sliced.max_data = None
                input_sliced.e_bias = None
            del input_unfold
            out = torch.cat(out_chunks, dim=1)
            del out_chunks

        # Compute output dimensions using saved kernel dims
        stride = self.stride if isinstance(self.stride, tuple) else (self.stride, self.stride)
        padding = self.padding if isinstance(self.padding, tuple) else (self.padding, self.padding)
        dilation = self.dilation if isinstance(self.dilation, tuple) else (self.dilation, self.dilation)
        h_out = int((input.shape[2] + 2 * padding[0] - dilation[0] * (self._kernel_h - 1) - 1) / stride[0] + 1)
        w_out = int((input.shape[3] + 2 * padding[1] - dilation[1] * (self._kernel_w - 1) - 1) / stride[1] + 1)

        # Async prefetch next streaming layer (overlaps with BN/ReLU/residual between layers)
        if self._next_streaming_layer is not None and self._next_streaming_layer._streaming:
            self._next_streaming_layer._async_prefetch(self.engine.device)

        # Release GPU tensors — point references back to pinned CPU buffers (NO D2H copy!)
        if self._streaming:
            self._release_gpu_tensors()

        # Cast output dtype to match input (RRAM engine computes in float32)
        if out.dtype != input.dtype:
            out = out.to(input.dtype)

        if self.bias is not None:
            out = out + self.bias
        out = F.fold(out.transpose(1, 2), output_size=(h_out, w_out), kernel_size=(1, 1))
        return out

    def _offload_to_cpu(self):
        """Initial offload: move G data from GPU to pinned CPU memory.

        Called ONCE during update_weight_and_prepare(streaming=True).
        Creates persistent pinned buffers in self._pinned_buffers for
        fast async prefetch during inference. These buffers are never freed
        and are reused on every forward pass.
        """
        ws = self.weight_sliced
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # ensure G computation is complete

        for attr in _STREAM_ATTRS:
            tensor = getattr(ws, attr, None)
            if tensor is not None:
                cpu_t = _to_cpu_pinned(tensor)
                self._pinned_buffers[attr] = cpu_t
                setattr(ws, attr, cpu_t)

    def _load_to_device(self, device):
        """Synchronous load: copy pinned CPU → GPU, blocks until complete.

        Used for the first streaming layer in a forward pass (no prefetch available).
        Subsequent layers should use _async_prefetch for overlap.
        """
        ws = self.weight_sliced
        for attr in _STREAM_ATTRS:
            pinned = self._pinned_buffers.get(attr)
            if pinned is not None:
                setattr(ws, attr, _to_device(pinned, device))
        torch.cuda.synchronize(device)

    def _async_prefetch(self, device):
        """Start loading G data to GPU on a dedicated transfer stream (non-blocking).

        Called by the PREVIOUS layer in the prefetch chain, so the H2D
        transfer overlaps with whatever computation happens between layers.
        """
        try:
            stream = Conv2dMem._get_transfer_stream(device)
            ws = self.weight_sliced
            with torch.cuda.stream(stream):
                for attr in _STREAM_ATTRS:
                    pinned = self._pinned_buffers.get(attr)
                    if pinned is not None:
                        setattr(ws, attr, _to_device(pinned, device))
            object.__setattr__(self, '_prefetch_event', stream.record_event())
        except torch.cuda.OutOfMemoryError:
            # Prefetch failed — reset to pinned CPU, next forward will sync-load
            self._release_gpu_tensors()
            object.__setattr__(self, '_prefetch_event', None)
            torch.cuda.empty_cache()

    def _release_gpu_tensors(self):
        """Release GPU tensors by pointing references back to pinned CPU buffers.

        KEY INSIGHT: G data never changes during inference, so we DON'T need
        D2H copies. We just set the SlicedData attributes back to the persistent
        pinned CPU tensors. The old GPU tensors become unreferenced and are freed.

        This eliminates 50% of the PCIe transfer overhead compared to the old
        approach (which did GPU→CPU copy + re-pin on every forward pass).
        """
        ws = self.weight_sliced
        for attr in _STREAM_ATTRS:
            pinned = self._pinned_buffers.get(attr)
            if pinned is not None:
                setattr(ws, attr, pinned)

    def update_weight(self):
        self.weight_sliced.slice_data_imp(self.engine, self.weight.reshape(self.weight.shape[0], -1).detach().t().to(self.engine.device))

def is_tuple_2(x):
    # if x is x tuple of 2 elements, return x, else return (x, x)
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    elif isinstance(x, int):
        return (x, x)
    else:
        raise ValueError("x must be x tuple or int")

def _test():
    torch.manual_seed(100)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X = torch.randn(5, 3, 96, 96, requires_grad=True, dtype=torch.float).to(device)
    engine = DPETensor(
        HGS=1e-5,                       # High conductance state
        LGS=1e-8,                       # Low conductance state
        write_variation=0.0,          # Write variation
        rate_stuck_HGS=0.001,          # Rate of stuck at HGS
        rate_stuck_LGS=0.000,          # Rate of stuck at LGS
        read_variation={0:0.05, 1:0.05, 2:0.05, 3:0.05},           # Read variation
        vnoise=0.0,                   # Random Gaussian noise of voltage
        rdac=2**2,                      # Number of DAC resolution
        g_level=2**2,                   # Number of conductance levels
        radc=2**12
        )
    xblk = [1, 1, 2, 2]
    mblk = [1, 1, 2, 2]

    layer = Conv2dMem(engine, 3, 6, 3, xblk, mblk, padding=1, stride=1, bias=False,
                      device=device, bw_e=None, input_paral_size=(1, 32), weight_paral_size=(32, 32),
                      input_quant_gran=(1, 64), weight_quant_gran=(64, 64))
    output = layer(X)
    output.backward(torch.ones_like(output))

    weight = layer.weight.data
    weight.requires_grad = True
    out = F.conv2d(X, weight, padding=1, stride=1)
    out.backward(torch.ones_like(out))

    print(torch.allclose(weight.grad, layer.weight.grad, atol=1e-4))
    print(weight.grad[0][0])
    print(layer.weight.grad[0][0])

if __name__== '__main__':
    _test()



