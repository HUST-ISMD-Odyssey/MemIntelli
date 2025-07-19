# -*- coding:utf-8 -*-
# @File  : vgg_cifar.py
# @Author: ZZW
# @Date  : 2025/02/20
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Dict, Any, cast, Optional
from torch.hub import load_state_dict_from_url
from memintelli.NN_layers import Conv2dMem, LinearMem

# Pretrained model URLs
cifar_pretrained_urls = {
    10: {
        'vgg11_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg11_bn-eaeebf42.pt',
        'vgg13_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg13_bn-c01e4a43.pt',
        'vgg16_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg16_bn-6ee7ea24.pt',
        'vgg19_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg19_bn-57191229.pt',
    },
    100: {
        'vgg11_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg11_bn-57d0759e.pt',
        'vgg13_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg13_bn-5ebe5778.pt',
        'vgg16_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg16_bn-7d8c4031.pt',
        'vgg19_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg19_bn-b98f7bd7.pt',
    }
}

# Configuration for different VGG architectures
cfgs: Dict[str, List[Union[str, int]]] = {
    'vgg11_bn': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13_bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16_bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19_bn': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_CIFAR(nn.Module):
    """
    Unified VGG model for CIFAR datasets with optional memristive mode.

    Args:
        cfg (str): Architecture configuration key
        num_classes (int): Number of output classes
        mem_enabled (bool): If mem_enabled is True, the model will use the memristive engine for memristive weight updates
        mem_args: Dictionary containing memristive parameters
    """
    def __init__(
        self,   
        cfg: str = 'vgg16_bn',
        num_classes: int = 10,
        mem_enabled: bool = True,
        mem_args: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.mem_enabled = mem_enabled
        self.mem_args = mem_args if self.mem_enabled else {}
        self.features = self._make_layers(cfgs[cfg])
        self.classifier = self._make_classifier(num_classes)

    def _make_layers(self, cfg: List[Union[str, int]]) -> nn.Sequential:
        """Construct feature extraction layers."""
        layers = []
        in_channels = 3

        for v in cfg:
            if v == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                # Choose Conv2d implementation based on mem_enabled
                conv_layer = Conv2dMem(**self.mem_args, in_channels=in_channels, out_channels=cast(int, v),
                kernel_size=3, padding=1) if self.mem_enabled else nn.Conv2d(in_channels, cast(int, v), kernel_size=3, padding=1)
                layers.extend([
                    conv_layer,
                    nn.BatchNorm2d(cast(int, v)),
                    nn.ReLU()
                ])
                in_channels = cast(int, v)

        return nn.Sequential(*layers)
    
    def _make_classifier(self, num_classes: int) -> nn.Sequential:
        """Construct classification head."""
        # Choose Linear implementation based on mem_enabled
        linear = LinearMem if self.mem_enabled else nn.Linear
        mem_args = self.mem_args if self.mem_enabled else {}

        return nn.Sequential(
            linear(in_features=512, out_features=512, **mem_args),
            nn.ReLU(),
            nn.Dropout(),
            linear(in_features=512, out_features=512, **mem_args),
            nn.ReLU(),
            nn.Dropout(),
            linear(in_features=512, out_features=num_classes, **mem_args)
        )

    def forward(self, x: Any) -> Any:
        """Forward pass implementation."""
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def update_weight(self) -> None:
        """Update weights for memory-efficient layers (if enabled)."""
        if not self.mem_enabled:
            return

        for module in self.modules():
            if isinstance(module, (LinearMem, Conv2dMem)):
                module.update_weight()

def vgg_cifar_zoo(
    model_name: str = 'vgg16_bn',
    num_classes: int = 10,
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
) -> VGG_CIFAR:
    """
    VGG model factory for CIFAR datasets.

    Args:
        model_name (str): Model architecture name
        num_classes (int): Number of output classes
        pretrained (bool): Load pretrained weights
        mem_enabled (bool): Enable memristive mode
        engine (Optional[Any]): Memory engine for Mem layers
        input_slice (Optional[torch.Tensor, list]): Input tensor slicing configuration
        weight_slice (Optional[torch.Tensor, list]): Weight tensor slicing configuration
        device (Optional[Any]): Computation device (CPU/GPU)
        bw_e (Optional[Any]): if bw_e is None, the memristive engine is INT mode, otherwise, the memristive engine is FP mode (bw_e is the bitwidth of the exponent)

    Returns:
        VGG_CIFAR: Configured VGG model instance
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
    
    model = VGG_CIFAR(
        cfg=model_name,
        num_classes=num_classes,
        mem_enabled=mem_enabled,
        mem_args=mem_args
    )

    if pretrained:
        url = cifar_pretrained_urls[num_classes][model_name]
        model.load_state_dict(load_state_dict_from_url(url))

    return model