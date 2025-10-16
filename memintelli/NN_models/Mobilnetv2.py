# -*- coding:utf-8 -*-
# @File  : Mobilnetv2.py
# @Author: Zhou
# @Date  : 2025/10/16

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Dict, Any, Optional
from torch.hub import load_state_dict_from_url
from memintelli.NN_layers import Conv2dMem, LinearMem

# Pretrained model URLs
model_urls = {
    'mobilenet_v2': 'https://github.com/HUST-ISMD-Odyssey/MemIntelli/releases/download/pretrained_model/mobilenetv2_1.0-0c6065bc.pth'
}


class InvertedResidual(nn.Module):
    """
    Inverted Residual block with optional memristive layers
    
    Args:
        inp (int): Number of input channels
        oup (int): Number of output channels
        stride (int): Stride for depthwise convolution
        expand_ratio (int): Expansion ratio for hidden dimension
        mem_enabled (bool): Enable memristive layers
        mem_args (Optional[Dict[str, Any]]): Dictionary containing memristive parameters
    """
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        mem_enabled: bool = False,
        mem_args: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        self.mem_enabled = mem_enabled
        self.mem_args = mem_args if self.mem_enabled else {}

        layers = []
        if expand_ratio != 1:
            # Pointwise expansion
            conv_layer = Conv2dMem if mem_enabled else nn.Conv2d
            layers.extend([
                conv_layer(
                    in_channels=inp, out_channels=hidden_dim, kernel_size=1,
                    stride=1, padding=0, bias=False, **self.mem_args
                ) if mem_enabled else nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        
        # Depthwise convolution - Note: memristive implementation uses standard conv with groups
        if mem_enabled:
            # For depthwise, we use standard nn.Conv2d with groups since Conv2dMem doesn't support groups efficiently
            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        else:
            layers.extend([
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        
        # Pointwise linear projection
        conv_layer = Conv2dMem if mem_enabled else nn.Conv2d
        layers.extend([
            conv_layer(
                in_channels=hidden_dim, out_channels=oup, kernel_size=1,
                stride=1, padding=0, bias=False, **self.mem_args
            ) if mem_enabled else nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """
    MobileNetV2 model with optional memristive mode
    
    Args:
        num_classes (int): Number of output classes
        width_mult (float): Width multiplier for channel dimensions
        inverted_residual_setting (Optional[List[List[int]]]): Network structure configuration
        round_nearest (int): Round the number of channels to the nearest multiple of this number
        mem_enabled (bool): Enable memristive layers
        mem_args (Optional[Dict[str, Any]]): Dictionary containing memristive parameters
    """
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        mem_enabled: bool = False,
        mem_args: Optional[Dict[str, Any]] = None
    ):
        super().__init__()

        self.mem_enabled = mem_enabled
        self.mem_args = mem_args if self.mem_enabled else {}

        block = InvertedResidual
        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            # t: expansion factor, c: output channels, n: number of blocks, s: stride
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # Only check the first element, assuming user knows what they are doing
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                           "or a 4-element list, got {}".format(inverted_residual_setting))

        # Building first layer
        input_channel = self._make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = self._make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        
        conv_layer = Conv2dMem if mem_enabled else nn.Conv2d
        features = [
            conv_layer(
                in_channels=3, out_channels=input_channel, kernel_size=3,
                stride=2, padding=1, bias=False, **self.mem_args
            ) if mem_enabled else nn.Conv2d(3, input_channel, 3, 2, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        ]

        # Building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = self._make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(
                    input_channel, output_channel, stride, expand_ratio=t,
                    mem_enabled=mem_enabled, mem_args=mem_args
                ))
                input_channel = output_channel

        # Building last several layers
        features.extend([
            conv_layer(
                in_channels=input_channel, out_channels=self.last_channel, kernel_size=1,
                stride=1, padding=0, bias=False, **self.mem_args
            ) if mem_enabled else nn.Conv2d(input_channel, self.last_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True),
        ])

        # Make it nn.Sequential
        self.features = nn.Sequential(*features)

        # Building classifier
        linear_layer = LinearMem if mem_enabled else nn.Linear
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            linear_layer(
                in_features=self.last_channel, out_features=num_classes,
                **self.mem_args
            ) if mem_enabled else nn.Linear(self.last_channel, num_classes),
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, Conv2dMem)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Linear, LinearMem)):
                nn.init.normal_(m.weight, 0, 0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _make_divisible(self, v: float, divisor: int, min_value: Optional[int] = None) -> int:
        """
        This function ensures that all layers have a channel number that is divisible by 8
        It can be seen at: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        """
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def update_weight(self) -> None:
        """Update weights for memristive layers (if enabled)"""
        if not self.mem_enabled:
            return

        for module in self.modules():
            if isinstance(module, (Conv2dMem, LinearMem)):
                module.update_weight()


def MobileNetV2_zoo(
    model_name: str = 'mobilenet_v2',
    num_classes: int = 1000,
    width_mult: float = 1.0,
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
) -> MobileNetV2:
    """
    MobileNetV2 model factory
    
    Args:
        model_name (str): Model architecture name (currently only 'mobilenet_v2')
        num_classes (int): Number of output classes
        width_mult (float): Width multiplier for channel dimensions (default: 1.0)
        pretrained (bool): Load pretrained weights
        mem_enabled (bool): Enable memristive mode
        engine (Optional[Any]): Memory engine for Mem layers
        input_slice (Optional[torch.Tensor, list]): Input tensor slicing configuration
        weight_slice (Optional[torch.Tensor, list]): Weight tensor slicing configuration
        device (Optional[Any]): Computation device (CPU/GPU)
        bw_e (Optional[Any]): If bw_e is None, the memristive engine is INT mode, 
                             otherwise, the memristive engine is FP mode (bw_e is the bitwidth of the exponent)
        input_paral_size (Optional[torch.Tensor, list]): Input parallelization size
        weight_paral_size (Optional[torch.Tensor, list]): Weight parallelization size
        input_quant_gran (Optional[torch.Tensor, list]): Input quantization granularity
        weight_quant_gran (Optional[torch.Tensor, list]): Weight quantization granularity
    
    Returns:
        MobileNetV2: Configured MobileNetV2 model
    
    Example:
        >>> # Standard PyTorch model
        >>> model = MobileNetV2_zoo('mobilenet_v2', num_classes=1000, pretrained=True)
        >>> 
        >>> # Memristive model
        >>> from memintelli.pimpy import MemIntelli
        >>> engine = MemIntelli(device='cuda')
        >>> model = MobileNetV2_zoo('mobilenet_v2', num_classes=1000, mem_enabled=True, engine=engine)
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

    if model_name not in model_urls:
        raise ValueError(f"Invalid model name: {model_name}. Available: {list(model_urls.keys())}")

    model = MobileNetV2(
        num_classes=num_classes,
        width_mult=width_mult,
        mem_enabled=mem_enabled,
        mem_args=mem_args
    )

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[model_name], progress=True)
        model.load_state_dict(state_dict)

    return model
