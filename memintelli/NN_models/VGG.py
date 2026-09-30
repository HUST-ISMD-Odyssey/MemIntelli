# -*- coding:utf-8 -*-
# @File  : VGG.py
# @Author: ZZW
# @Date  : 2025/02/20

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Dict, Any, cast, Optional
from memintelli.NN_layers import Conv2dMem, LinearMem, tensor_bytes

# timm model names on HuggingFace (timm/xxx)
timm_model_names: Dict[str, str] = {
    'vgg11': 'vgg11.tv_in1k',
    'vgg13': 'vgg13.tv_in1k',
    'vgg16': 'vgg16.tv_in1k',
    'vgg19': 'vgg19.tv_in1k',
    'vgg11_bn': 'vgg11_bn.tv_in1k',
    'vgg13_bn': 'vgg13_bn.tv_in1k',
    'vgg16_bn': 'vgg16_bn.tv_in1k',
    'vgg19_bn': 'vgg19_bn.tv_in1k',
}


def _load_timm_pretrained(model: nn.Module, model_name: str) -> None:
    """Load pretrained weights from timm (HuggingFace: timm/xxx).

    timm VGG uses Conv2d-based ConvMlp for pre_logits and ClassifierHead for head,
    so we need to remap keys and reshape classifier weights (Conv2d → Linear).

    timm state_dict key mapping:
        pre_logits.fc1  →  classifier.0   (Conv2d [out, in, 7, 7] → Linear [out, in*7*7])
        pre_logits.fc2  →  classifier.3   (Conv2d [out, in, 1, 1] → Linear [out, in])
        head.fc         →  classifier.6   (Linear, same shape)
        features.*      →  features.*     (same)
    """
    import timm

    timm_name = timm_model_names[model_name]
    timm_model = timm.create_model(timm_name, pretrained=True)
    timm_sd = timm_model.state_dict()

    new_sd = {}
    for k, v in timm_sd.items():
        if k.startswith('features.'):
            new_sd[k] = v
        elif k.startswith('pre_logits.fc1'):
            new_key = k.replace('pre_logits.fc1', 'classifier.0')
            if 'weight' in k:
                # Conv2d weight [out_ch, in_ch, 7, 7] → Linear weight [out_ch, in_ch*7*7]
                v = v.reshape(v.shape[0], -1)
            new_sd[new_key] = v
        elif k.startswith('pre_logits.fc2'):
            new_key = k.replace('pre_logits.fc2', 'classifier.3')
            if 'weight' in k:
                # Conv2d weight [out_ch, in_ch, 1, 1] → Linear weight [out_ch, in_ch]
                v = v.reshape(v.shape[0], -1)
            new_sd[new_key] = v
        elif k.startswith('head.fc'):
            new_key = k.replace('head.fc', 'classifier.6')
            new_sd[new_key] = v
        # skip other timm-specific keys (e.g. head.flatten, etc.)

    model.load_state_dict(new_sd)
    del timm_model, timm_sd
    print(f"[VGG] Loaded pretrained weights from timm/{timm_name}")

# Configuration for different VGG architectures
cfgs: Dict[str, List[Union[str, int]]] = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    """
    Unified VGG model for ImageNet with optional memristive mode.

    Args:
        cfg (str): Architecture configuration key (e.g. 'vgg16', 'vgg16_bn')
        num_classes (int): Number of output classes
        batch_norm (bool): Whether to use batch normalization
        mem_enabled (bool): If True, use memristive engine layers
        mem_args: Dictionary containing memristive parameters
    """
    def __init__(
        self,
        cfg: str = 'vgg16',
        num_classes: int = 1000,
        batch_norm: bool = False,
        mem_enabled: bool = True,
        mem_args: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.mem_enabled = mem_enabled
        self.mem_args = mem_args if self.mem_enabled else {}
        self.batch_norm = batch_norm

        self.features = self._make_layers(cfgs[cfg])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = self._make_classifier(num_classes)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, Conv2dMem)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.Linear, LinearMem)):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, cfg: List[Union[str, int]]) -> nn.Sequential:
        """Construct feature extraction layers."""
        layers = []
        in_channels = 3

        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                v = cast(int, v)
                if self.mem_enabled:
                    conv_layer = Conv2dMem(
                        **self.mem_args, in_channels=in_channels, out_channels=v,
                        kernel_size=3, padding=1, skip_initial_mapping=True
                    )
                else:
                    conv_layer = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)

                if self.batch_norm:
                    layers.extend([conv_layer, nn.BatchNorm2d(v), nn.ReLU(inplace=True)])
                else:
                    layers.extend([conv_layer, nn.ReLU(inplace=True)])

                in_channels = v

        return nn.Sequential(*layers)

    def _make_classifier(self, num_classes: int) -> nn.Sequential:
        """Construct classification head (ImageNet-style with 4096 hidden units)."""
        linear = LinearMem if self.mem_enabled else nn.Linear
        mem_args = self.mem_args if self.mem_enabled else {}
        # Add skip_initial_mapping for LinearMem to avoid computing G from random weights
        if self.mem_enabled:
            mem_args = {**mem_args, 'skip_initial_mapping': True}

        return nn.Sequential(
            linear(in_features=512 * 7 * 7, out_features=4096, **mem_args),
            nn.ReLU(True),
            nn.Dropout(),
            linear(in_features=4096, out_features=4096, **mem_args),
            nn.ReLU(True),
            nn.Dropout(),
            linear(in_features=4096, out_features=num_classes, **mem_args),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass implementation."""
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def update_weight(self) -> None:
        """Update weights for memristive layers (if enabled)."""
        if not self.mem_enabled:
            return

        for module in self.modules():
            if isinstance(module, (Conv2dMem, LinearMem)):
                module.update_weight()

    def prepare_for_inference(self) -> None:
        """Prepare the model for optimized inference.

        Call after update_weight() and before inference.
        For lower peak memory, use update_weight_and_prepare() instead.
        """
        self.eval()
        if not self.mem_enabled:
            return

        import gc
        for module in self.modules():
            if isinstance(module, (Conv2dMem, LinearMem)):
                module.inference_mode = True
                module.weight_sliced.inference = True
                engine = module.engine
                if engine.write_variation == 0:
                    module.weight_sliced.compress_G(engine)
                module.weight_sliced.quantized_data = None
                module.weight_sliced.sliced_data = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def update_weight_and_prepare(self, streaming=False, free_weights: bool = True,
                                   gpu_memory_reserve: float = 4.0) -> None:
        """Combined update + prepare that minimizes peak GPU memory.

        Architecture: 3-phase pipeline (same as Qwen3).

        Phase 1: For each layer sequentially:
          weight → GPU → compute G → compress → offload G to pinned CPU → free weight
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
        all_mem_layers = []
        layer_count = 0
        engine_device = None
        for module in self.modules():
            if isinstance(module, (Conv2dMem, LinearMem)):
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
                module._offload_to_cpu()

                all_mem_layers.append(module)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not all_mem_layers:
            return

        # ─── Phase 2: Decide streaming strategy and load GPU-resident layers ───
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
            print(f"[VGG] {auto_mode}: GPU {gpu_total/1024**3:.1f}GB total, "
                  f"{gpu_used/1024**3:.1f}GB used, "
                  f"{pending_gb:.1f}GB pending, "
                  f"{gpu_memory_reserve:.0f}GB reserved → "
                  f"budget {budget_gb:.1f}GB for G")

            if total_g_bytes <= gpu_budget:
                streaming = False
                print(f"[VGG] {auto_mode}: ALL G ({total_g_gb:.1f}GB) fits → GPU-resident (fastest)")
            else:
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

                if offload_set:
                    max_stream_size = max(layer_g_sizes[i] for i in offload_set)
                    resident_bytes = total_g_bytes - offloaded_bytes
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

                for i, m in enumerate(all_mem_layers):
                    if i in offload_set:
                        m._streaming = True
                    else:
                        m._load_to_device(engine_device)
                        m._pinned_buffers.clear()

                resident_count = layer_count - len(offload_set)
                resident_gb = (total_g_bytes - offloaded_bytes) / 1024**3
                print(f"[VGG] {auto_mode}: partial: {resident_count} GPU-resident ({resident_gb:.1f}GB), "
                      f"{len(offload_set)} streaming ({offloaded_bytes/1024**3:.1f}GB)")
                streaming = "partial_done"

        if streaming is False:
            for m in all_mem_layers:
                m._load_to_device(engine_device)
                m._pinned_buffers.clear()
            print(f"[VGG] Processed {layer_count} layers → GPU-resident "
                  f"({total_g_gb:.1f}GB G on GPU, weights {'freed' if free_weights else 'kept'})")
        elif streaming is True:
            for m in all_mem_layers:
                m._streaming = True
            print(f"[VGG] Processed {layer_count} layers → full streaming "
                  f"({total_g_gb:.1f}GB G on CPU, weights {'freed' if free_weights else 'kept'})")

        # ─── Phase 3: Build async prefetch chain for streaming layers ───
        streaming_layers = [m for m in all_mem_layers if m._streaming]
        if streaming_layers:
            for i in range(len(streaming_layers) - 1):
                object.__setattr__(streaming_layers[i], '_next_streaming_layer', streaming_layers[i + 1])
            object.__setattr__(streaming_layers[-1], '_next_streaming_layer', streaming_layers[0])
            print(f"[VGG] Async prefetch chain: {len(streaming_layers)} streaming layers linked")


def VGG_zoo(
    model_name: str = 'vgg16',
    num_classes: int = 1000,
    pretrained: bool = False,
    mem_enabled: bool = False,
    engine: Optional[Any] = None,
    input_slice: Optional[Union[torch.Tensor, list]] = [1, 1, 2, 4],
    weight_slice: Optional[Union[torch.Tensor, list]] = [1, 1, 2, 4],
    device: Optional[Any] = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    bw_e: Optional[Any] = None,
    input_paral_size: Optional[Union[torch.Tensor, list]] = (1, 32),
    weight_paral_size: Optional[Union[torch.Tensor, list]] = (32, 32),
    input_quant_gran: Optional[Union[torch.Tensor, list]] = (1, 64),
    weight_quant_gran: Optional[Union[torch.Tensor, list]] = (64, 64)
) -> VGG:
    """
    VGG model factory for ImageNet.

    Args:
        model_name (str): Model architecture name (e.g. 'vgg16', 'vgg16_bn')
        num_classes (int): Number of output classes
        pretrained (bool): Load pretrained weights
        mem_enabled (bool): Enable memristive mode
        engine (Optional[Any]): Memory engine for Mem layers
        input_slice (Optional[torch.Tensor, list]): Input tensor slicing configuration
        weight_slice (Optional[torch.Tensor, list]): Weight tensor slicing configuration
        device (Optional[Any]): Computation device (CPU/GPU)
        bw_e (Optional[Any]): if bw_e is None, the memristive engine is INT mode,
            otherwise, the memristive engine is FP mode (bw_e is the bitwidth of the exponent)

    Returns:
        VGG: Configured VGG model instance
    """
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
    } if mem_enabled else {}

    if model_name not in timm_model_names:
        raise ValueError(f"Invalid model name: {model_name}. "
                         f"Choose from {list(timm_model_names.keys())}")

    # Determine base config and whether to use batch norm
    batch_norm = model_name.endswith('_bn')
    base_cfg = model_name.replace('_bn', '') if batch_norm else model_name

    model = VGG(
        cfg=base_cfg,
        num_classes=num_classes,
        batch_norm=batch_norm,
        mem_enabled=mem_enabled,
        mem_args=mem_args
    )

    if pretrained:
        _load_timm_pretrained(model, model_name)

    return model
