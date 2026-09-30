# -*- coding:utf-8 -*-
# @File  : ResNet.py
# @Author: Zhou
# @Date  : 2024/4/1

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Dict, Any, cast, Optional, Type
from torch.hub import load_state_dict_from_url
from memintelli.NN_layers import Conv2dMem, LinearMem, tensor_bytes

resnet_pretrained_urls = {
    10: {
        'resnet18': 'https://github.com/HUST-ISMD-Odyssey/MemIntelli/releases/download/pretrained_model/resnet18-cifar10_9498.bin'
    },
    100: {
        'resnet18': 'https://github.com/HUST-ISMD-Odyssey/MemIntelli/releases/download/pretrained_model/resnet18-cifar100_7926.bin'
    }
}

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        mem_enabled: bool = False,
        mem_args: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.mem_enabled = mem_enabled
        self.mem_args = mem_args if self.mem_enabled else {}

        conv_layer = Conv2dMem if self.mem_enabled else nn.Conv2d

        self.conv1 = conv_layer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False, **self.mem_args
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv_layer(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3,
            stride=1, padding=1, bias=False, **self.mem_args
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        mem_enabled: bool = False,
        mem_args: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.mem_enabled = mem_enabled
        self.mem_args = mem_args if self.mem_enabled else {}

        conv_layer = Conv2dMem if self.mem_enabled else nn.Conv2d

        self.conv1 = conv_layer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1,
            stride=1, bias=False, **self.mem_args
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv_layer(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False, **self.mem_args
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv_layer(
            in_channels=out_channels, out_channels=out_channels*self.expansion, kernel_size=1,
            stride=1, bias=False, **self.mem_args
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet_CIFAR(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 100,
        mem_enabled: bool = False,
        mem_args: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.mem_enabled = mem_enabled
        self.mem_args = mem_args if self.mem_enabled else {}
        self.in_channels = 64
        conv_layer = Conv2dMem if self.mem_enabled else nn.Conv2d
        self.conv1 = conv_layer(
            in_channels=3, out_channels=64, kernel_size=3,
            stride=1, bias=False, **self.mem_args
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        linear_layer = LinearMem if self.mem_enabled else nn.Linear
        self.fc = linear_layer(
            in_features=512 * block.expansion, out_features=num_classes, **self.mem_args
        )
        # Weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        conv_layer = Conv2dMem if self.mem_enabled else nn.Conv2d

        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                conv_layer(
                    in_channels=self.in_channels, out_channels=channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False, **self.mem_args
                ),
                nn.BatchNorm2d(channels * block.expansion),
            )

        layers = []
        layers.append(block(
            self.in_channels, channels,
            stride=stride,
            downsample=downsample,
            mem_enabled=self.mem_enabled,
            mem_args=self.mem_args
        ))
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(
                self.in_channels, channels,
                mem_enabled=self.mem_enabled,
                mem_args=self.mem_args
            ))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def update_weight(self) -> None:
        """Update weights for memory-efficient layers (if enabled)."""
        if not self.mem_enabled:
            return

        for module in self.modules():
            if isinstance(module, (LinearMem, Conv2dMem)):
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
            print(f"[ResNet_CIFAR] {auto_mode}: GPU {gpu_total/1024**3:.1f}GB total, "
                  f"{gpu_used/1024**3:.1f}GB used, "
                  f"{gpu_memory_reserve:.0f}GB reserved → "
                  f"budget {budget_gb:.1f}GB for G")

            if total_g_bytes <= gpu_budget:
                streaming = False
                print(f"[ResNet_CIFAR] {auto_mode}: ALL G ({total_g_gb:.1f}GB) fits → GPU-resident")
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
                print(f"[ResNet_CIFAR] {auto_mode}: partial: {resident_count} GPU-resident, "
                      f"{len(offload_set)} streaming")
                streaming = "partial_done"

        if streaming is False:
            for m in all_mem_layers:
                m._load_to_device(engine_device)
                m._pinned_buffers.clear()
            print(f"[ResNet_CIFAR] {layer_count} layers → GPU-resident ({total_g_gb:.1f}GB G)")
        elif streaming is True:
            for m in all_mem_layers:
                m._streaming = True
            print(f"[ResNet_CIFAR] {layer_count} layers → streaming ({total_g_gb:.1f}GB G on CPU)")

        # ─── Phase 3: Build async prefetch chain for streaming layers ───
        streaming_layers = [m for m in all_mem_layers if m._streaming]
        if streaming_layers:
            for i in range(len(streaming_layers) - 1):
                object.__setattr__(streaming_layers[i], '_next_streaming_layer', streaming_layers[i + 1])
            object.__setattr__(streaming_layers[-1], '_next_streaming_layer', streaming_layers[0])
            print(f"[ResNet_CIFAR] Async prefetch chain: {len(streaming_layers)} layers linked")
def ResNet_CIFAR_zoo(
    model_name: str = 'resnet18',
    num_classes: int = 100,
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
) -> ResNet_CIFAR:

    # Architecture configuration
    model_params: Dict[str, Any] = {
        'resnet18': (BasicBlock, [2, 2, 2, 2]),
        'resnet34': (BasicBlock, [3, 4, 6, 3]),
        'resnet50': (Bottleneck, [3, 4, 6, 3]),
        'resnet101': (Bottleneck, [3, 4, 23, 3]),
        'resnet152': (Bottleneck, [3, 8, 36, 3])
    }

    mem_args = {
        "engine": engine,
        "input_slice": input_slice,
        "weight_slice": weight_slice,
        "device": device,
        "bw_e": bw_e,
        "input_paral_size": input_paral_size,
        "weight_paral_size": weight_paral_size,
        "input_quant_gran": input_quant_gran,
        "weight_quant_gran": weight_quant_gran,
        "skip_initial_mapping": True,
    } if mem_enabled else {}

    if model_name not in model_params:
        raise ValueError(f"Invalid model name: {model_name}")

    block, layers = model_params[model_name]
    model = ResNet_CIFAR(
        block=block,
        layers=layers,
        num_classes=num_classes,
        mem_enabled=mem_enabled,
        mem_args=mem_args
    )

    if pretrained:
        url = resnet_pretrained_urls[num_classes][model_name]
        model.load_state_dict(load_state_dict_from_url(url))

    return model
