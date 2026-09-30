# -*- coding:utf-8 -*-
# @File  : Qwen3.py
# @Author: Zhou
# @Date  : 2026/2/13

from typing import Optional, Union, Any

import torch
import torch.nn as nn

from memintelli.NN_layers import LinearMem, tensor_bytes

try:
    from transformers import AutoModelForCausalLM
except ImportError as exc:
    AutoModelForCausalLM = None
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None


class Qwen3MemWrapper(nn.Module):
    """Qwen3 wrapper with optional memristive linear layers."""

    def __init__(self, model: nn.Module, mem_enabled: bool = False):
        super().__init__()
        self.model = model
        self.mem_enabled = mem_enabled

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def update_weight(self):
        if not self.mem_enabled:
            return
        for module in self.modules():
            if isinstance(module, LinearMem):
                module.update_weight()

    def prepare_for_inference(self) -> None:
        """Prepare the model for optimized inference.

        This method:
        1. Sets eval mode (BatchNorm, Dropout, etc.)
        2. Enables inference_mode on all memristive layers (bypasses autograd)
        3. Compresses G to uint8 indices if possible (~4x memory savings)
        4. Frees training-only data (quantized_data, sliced_data) from weight SlicedData

        Call this after update_weight() and before inference.
        For lower peak memory during initialization, use update_weight_and_prepare() instead.
        """
        self.eval()  # set BatchNorm, Dropout to eval mode

        if not self.mem_enabled:
            return

        import gc
        for module in self.modules():
            if isinstance(module, LinearMem):
                # Enable inference mode on the layer
                module.inference_mode = True
                # Mark weight SlicedData as inference (for engine routing)
                module.weight_sliced.inference = True
                # Compress G to uint8 indices (~4x memory savings)
                # Valid when write_variation=0 (G values are exactly on conductance levels)
                engine = module.engine
                if engine.write_variation == 0:
                    module.weight_sliced.compress_G(engine)
                # Free training-only data from weight
                module.weight_sliced.quantized_data = None
                module.weight_sliced.sliced_data = None

        # Force garbage collection to reclaim freed memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def update_weight_and_prepare(self, streaming=False, free_weights: bool = True,
                                   gpu_memory_reserve: float = 4.0) -> None:
        """Combined update + prepare that minimizes peak GPU memory.

        Architecture: 3-phase pipeline that NEVER accumulates G on GPU.

        Phase 1: For each layer sequentially:
          weight(CPU) → GPU → compute G → compress → offload G to pinned CPU → free weight
          Peak GPU = ONE layer's (weight + G + intermediates) at any time.

        Phase 2: Decide strategy and selectively load G back to GPU:
          - False:  load ALL G back to GPU (fastest inference)
          - True:   keep ALL on CPU, stream per-layer (lowest memory)
          - "auto": load as many as GPU allows, stream the rest (best tradeoff)

        Phase 3: Build async prefetch chain for streaming layers.

        Args:
            streaming: False | True | "auto" (see above)
            free_weights: If True (default), free nn.Parameter weight data after G.
            gpu_memory_reserve: GB reserved for activations when streaming="auto".
        """
        self.eval()
        if not self.mem_enabled:
            return

        import gc

        # ─── Phase 1: Compute G for all layers, ALWAYS offload to CPU immediately ───
        # This keeps peak GPU = ONE layer at a time, regardless of model size.
        all_mem_layers = []
        layer_count = 0
        engine_device = None
        for module in self.modules():
            if isinstance(module, LinearMem):
                layer_count += 1
                engine = module.engine
                engine_device = engine.device

                module.weight_sliced.inference = True
                module.inference_mode = True
                module.update_weight()

                if engine.write_variation == 0:
                    module.weight_sliced.compress_G(engine)

                if free_weights:
                    module.weight.data = torch.empty(0, device='cpu', dtype=module.weight.dtype)

                module.weight_sliced.quantized_data = None
                module.weight_sliced.sliced_data = None

                if module.bias is not None and module.bias.device != engine_device:
                    module.bias.data = module.bias.data.to(engine_device)
                if module.input_slice_method.device != engine_device:
                    module.input_slice_method = module.input_slice_method.to(engine_device)
                if module.weight_slice_method.device != engine_device:
                    module.weight_slice_method = module.weight_slice_method.to(engine_device)

                # CRITICAL: Always offload G to pinned CPU immediately after computing!
                # Without this, G data from all processed layers would accumulate on GPU
                # and cause OOM for models with many layers (e.g., 3B+ params).
                module._offload_to_cpu()

                all_mem_layers.append(module)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not all_mem_layers:
            return

        # ─── Phase 2: Decide streaming strategy and load GPU-resident layers ───
        # All G is now on pinned CPU. Calculate sizes from pinned buffers.
        layer_g_sizes = []
        total_g_bytes = 0
        for m in all_mem_layers:
            sz = 0
            for attr in ('G_indices', 'G', 'max_data', 'e_bias'):
                t = m._pinned_buffers.get(attr)
                sz += tensor_bytes(t)
            layer_g_sizes.append(sz)
            total_g_bytes += sz

        total_g_gb = total_g_bytes / 1024**3

        auto_mode = "auto_memory" if streaming == "auto" else streaming
        if auto_mode in ("auto_memory", "auto_speed") and torch.cuda.is_available():
            gpu_total = torch.cuda.get_device_properties(engine_device).total_memory
            gpu_used = torch.cuda.memory_allocated(engine_device)

            # Account for non-LinearMem params that model.to(device) will add later
            # (embedding, layer norms, lm_head if not simulated, etc.)
            # After Phase 1: LinearMem weights are freed (numel=0), biases already on GPU.
            # So any CPU param with numel>0 is a non-LinearMem param that will be moved.
            pending_cpu_bytes = sum(
                p.numel() * p.element_size()
                for p in self.parameters()
                if p.device.type == 'cpu' and p.numel() > 0
            )

            gpu_budget = (gpu_total - gpu_used
                          - int(gpu_memory_reserve * 1024**3)
                          - pending_cpu_bytes)
            budget_gb = gpu_budget / 1024**3
            pending_gb = pending_cpu_bytes / 1024**3
            print(f"[Qwen3] {auto_mode}: GPU {gpu_total/1024**3:.1f}GB total, "
                  f"{gpu_used/1024**3:.1f}GB used, "
                  f"{pending_gb:.1f}GB pending (embedding/norms), "
                  f"{gpu_memory_reserve:.0f}GB reserved → "
                  f"budget {budget_gb:.1f}GB for G")

            if total_g_bytes <= gpu_budget:
                # All G fits — load everything back to GPU
                streaming = False
                print(f"[Qwen3] {auto_mode}: ALL G ({total_g_gb:.1f}GB) fits in GPU "
                      f"(budget {budget_gb:.1f}GB) → GPU-resident (fastest)")
            else:
                # auto_memory: offload biggest layers first (maximize memory safety).
                # auto_speed: offload smallest layers first (minimize transfer overhead).
                indexed = sorted(
                    range(len(layer_g_sizes)),
                    key=lambda i: layer_g_sizes[i],
                    reverse=(auto_mode == "auto_memory"),
                )
                need_to_offload = max(0, total_g_bytes - gpu_budget)
                offloaded_bytes = 0
                offload_set = set()

                for idx in indexed:
                    if offloaded_bytes >= need_to_offload:
                        break
                    offload_set.add(idx)
                    offloaded_bytes += layer_g_sizes[idx]

                # Post-check: ensure GPU has room for async prefetch double-buffer.
                # During inference, the current streaming layer's G is on GPU while
                # the next streaming layer is being prefetched — both coexist briefly.
                # We must reserve room for the LARGEST streaming layer as prefetch buffer.
                if offload_set:
                    max_stream_size = max(layer_g_sizes[i] for i in offload_set)
                    resident_bytes = total_g_bytes - offloaded_bytes
                    # Keep offloading according to the selected auto policy until prefetch fits.
                    while resident_bytes + max_stream_size > gpu_budget:
                        resident_indices = [i for i in range(len(all_mem_layers))
                                            if i not in offload_set]
                        if not resident_indices:
                            break
                        selected = (
                            max(resident_indices, key=lambda i: layer_g_sizes[i])
                            if auto_mode == "auto_memory"
                            else min(resident_indices, key=lambda i: layer_g_sizes[i])
                        )
                        offload_set.add(selected)
                        offloaded_bytes += layer_g_sizes[selected]
                        resident_bytes -= layer_g_sizes[selected]
                        max_stream_size = max(layer_g_sizes[i] for i in offload_set)

                # Load GPU-resident layers, mark streaming layers
                for i, m in enumerate(all_mem_layers):
                    if i in offload_set:
                        m._streaming = True
                        # G already on pinned CPU, _pinned_buffers ready
                    else:
                        m._load_to_device(engine_device)
                        m._pinned_buffers.clear()  # free pinned CPU copy

                resident_count = layer_count - len(offload_set)
                resident_gb = (total_g_bytes - offloaded_bytes) / 1024**3
                print(f"[Qwen3] {auto_mode}: G {total_g_gb:.1f}GB > budget {budget_gb:.1f}GB → "
                      f"partial: {resident_count} GPU-resident ({resident_gb:.1f}GB), "
                      f"{len(offload_set)} streaming ({offloaded_bytes/1024**3:.1f}GB)")
                streaming = "partial_done"

        if streaming is False:
            # Load ALL G back from pinned CPU to GPU
            for m in all_mem_layers:
                m._load_to_device(engine_device)
                m._pinned_buffers.clear()
            print(f"[Qwen3] Processed {layer_count} layers → GPU-resident "
                  f"({total_g_gb:.1f}GB G on GPU, weights {'freed' if free_weights else 'kept'})")
        elif streaming is True:
            # All stay on pinned CPU
            for m in all_mem_layers:
                m._streaming = True
            print(f"[Qwen3] Processed {layer_count} layers → full streaming "
                  f"({total_g_gb:.1f}GB G on CPU, weights {'freed' if free_weights else 'kept'})")
        # "partial_done" case already printed above

        # ─── Phase 3: Build async prefetch chain for streaming layers ───
        # Each streaming layer's forward() triggers async H2D for the next one,
        # overlapping PCIe transfers with GPU computation (attention, norms, etc.)
        streaming_layers = [m for m in all_mem_layers if m._streaming]
        if streaming_layers:
            for i in range(len(streaming_layers) - 1):
                # MUST use object.__setattr__ to bypass nn.Module.__setattr__!
                # PyTorch auto-registers Module attributes as submodules, which would
                # create a circular module tree and cause RecursionError in .to()
                object.__setattr__(streaming_layers[i], '_next_streaming_layer', streaming_layers[i + 1])
            # Circular: last layer prefetches first for next forward pass
            object.__setattr__(streaming_layers[-1], '_next_streaming_layer', streaming_layers[0])
            print(f"[Qwen3] Async prefetch chain: {len(streaming_layers)} streaming layers linked")

    def enable_streaming(self) -> None:
        """Enable streaming mode: offload all layer G data to CPU, load on-demand.

        Call this after update_weight_and_prepare() to further reduce GPU memory.
        During inference, each LinearMem layer will:
          1. Load its G_indices/G + max_data from CPU → GPU
          2. Perform the memristive dot product
          3. Offload G_indices/G + max_data back to CPU

        This means only ONE layer's G data is on GPU at any time.

        Memory savings example (Qwen3-0.6B, 4 slices, write_variation=0):
          - Without streaming: ~2GB GPU for all compressed G_indices
          - With streaming: ~100MB GPU (largest single layer's G_indices)

        Speed tradeoff: ~5-15ms overhead per layer per forward pass (PCIe transfer).
        """
        if not self.mem_enabled:
            return
        for module in self.modules():
            if isinstance(module, LinearMem):
                module._streaming = True
                module._offload_to_cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def config(self):
        return self.model.config

    def generate(self, *args, **kwargs):
        return self.model.generate(*args, **kwargs)


def _replace_linear_with_linearmem(module: nn.Module, mem_args: dict,
                                    skip_modules: set = None, prefix: str = ""):
    """Recursively replace nn.Linear with LinearMem, skipping specified modules.

    MEMORY OPTIMIZATION: Creates LinearMem on CPU with the original model dtype
    (e.g. bfloat16) instead of GPU float32. This prevents GPU memory doubling
    during replacement. For 8B models: avoids 16GB(old) + 32GB(new) = 48GB peak.

    The actual weight→conductance mapping happens later in update_weight_and_prepare(),
    which processes ONE layer at a time to minimize peak GPU memory.
    """
    import gc
    for name, child in list(module.named_children()):
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            if skip_modules and full_name in skip_modules:
                continue  # keep as regular nn.Linear
            # Create on CPU with original dtype to avoid GPU memory explosion.
            # For 8B bfloat16: this prevents trying to hold 48GB on a 24GB GPU.
            new_layer = LinearMem(
                engine=mem_args["engine"],
                in_features=child.in_features,
                out_features=child.out_features,
                input_slice=mem_args["input_slice"],
                weight_slice=mem_args["weight_slice"],
                bias=child.bias is not None,
                device='cpu',  # CPU → avoids GPU memory doubling
                dtype=child.weight.dtype,  # match original (bfloat16, not float32)
                bw_e=mem_args["bw_e"],
                input_paral_size=mem_args["input_paral_size"],
                weight_paral_size=mem_args["weight_paral_size"],
                input_quant_gran=mem_args["input_quant_gran"],
                weight_quant_gran=mem_args["weight_quant_gran"],
                skip_initial_mapping=True,
            )
            with torch.no_grad():
                new_layer.weight.copy_(child.weight.data)
                if child.bias is not None:
                    new_layer.bias.copy_(child.bias.data)
            setattr(module, name, new_layer)
            del child  # explicitly free old layer (releases GPU weight memory)
            gc.collect()
        else:
            _replace_linear_with_linearmem(child, mem_args, skip_modules, full_name)


def qwen3_zoo(
    model_name: str = "Qwen/Qwen3-0.6B",
    pretrained: bool = True,
    mem_enabled: bool = False,
    engine: Optional[Any] = None,
    input_slice: Optional[Union[torch.Tensor, list]] = [1, 1, 2, 4],
    weight_slice: Optional[Union[torch.Tensor, list]] = [1, 1, 2, 4],
    device: Optional[Any] = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    bw_e: Optional[Any] = None,
    input_paral_size: Optional[Union[torch.Tensor, list]] = (1, 32),
    weight_paral_size: Optional[Union[torch.Tensor, list]] = (32, 32),
    input_quant_gran: Optional[Union[torch.Tensor, list]] = (1, 64),
    weight_quant_gran: Optional[Union[torch.Tensor, list]] = (64, 64),
    torch_dtype: Optional[torch.dtype] = None,
    dtype: Optional[torch.dtype] = None,
    trust_remote_code: bool = True,
    skip_embedding_and_head: bool = True,
) -> Qwen3MemWrapper:
    """Qwen3 model factory with optional linear-to-LinearMem conversion.

    Args:
        skip_embedding_and_head: If True (default), keep embedding and lm_head as
            standard layers (no RRAM simulation). Highly recommended because:
              - lm_head is the LARGEST linear layer (e.g. 2560→151936 for Qwen3-4B)
              - Its G_indices alone occupy ~1.5GB, more than all other layers combined
              - RRAM simulation for lm_head takes ~50% of total inference time
              - Skipping it roughly HALVES total inference time and saves huge memory
              - lm_head is a simple projection; RRAM non-idealities there rarely matter
              - Embedding (nn.Embedding) is never replaced regardless of this flag
            Set to False to simulate ALL linear layers including lm_head.
    """
    if AutoModelForCausalLM is None:
        raise ImportError(
            "transformers is required for qwen3_zoo. "
            "Please install it via `pip install transformers`."
        ) from _TRANSFORMERS_IMPORT_ERROR

    if not pretrained:
        raise ValueError("qwen3_zoo currently supports pretrained=True only.")

    # Build skip set from the bool flag
    skip_modules = {"lm_head"} if skip_embedding_and_head else set()

    _dtype = dtype if dtype is not None else torch_dtype

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=_dtype,
        trust_remote_code=trust_remote_code,
    )

    if mem_enabled:
        if engine is None:
            raise ValueError("`engine` must be provided when mem_enabled=True.")
        mem_args = {
            "engine": engine,
            "input_slice": input_slice,
            "weight_slice": weight_slice,
            "device": device,
            "bw_e": bw_e,
            "input_paral_size": input_paral_size,
            "weight_paral_size": weight_paral_size,
            "input_quant_gran": input_quant_gran,
            "weight_quant_gran": weight_quant_gran
        }
        _replace_linear_with_linearmem(model, mem_args, skip_modules=skip_modules)
        if skip_modules:
            print(f" Skipped RRAM simulation for: {skip_modules} (using standard nn.Linear)")

    return Qwen3MemWrapper(model=model, mem_enabled=mem_enabled)
