# -*- coding:utf-8 -*-
# @File  : ResNet.py
# @Author: Zhou
# @Date  : 2024/4/1

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from typing import Union, List, Dict, Any, cast, Optional, Type
import os
import sys
# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from NN_layers import Conv2dMem, LinearMem, SliceMethod

# Pretrained model URLs
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class BasicBlock(nn.Module):
    """
    Basic residual block with optional memristive layers
    
    Args:
        mem_enabled: Enable memristive layers
        mem_args: Dictionary containing memristive parameters
    """
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

        # Choose convolutional layer type
        conv_layer = Conv2dMem if mem_enabled else nn.Conv2d

        self.conv1 = conv_layer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False, **mem_args
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv_layer(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3,
            stride=1, padding=1, bias=False, **mem_args
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
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
    """Bottleneck residual block with optional memristive layers"""
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

        conv_layer = Conv2dMem if mem_enabled else nn.Conv2d

        self.conv1 = conv_layer(
            in_channels=in_channels, out_channels=out_channels, kernel_size=1,
            stride=1, bias=False, **mem_args
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv_layer(
            in_channels=out_channels, out_channels=out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False, **mem_args
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = conv_layer(
            in_channels=out_channels, out_channels=out_channels*self.expansion, kernel_size=1,
            stride=1, bias=False, **mem_args
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
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

class ResNet(nn.Module):
    """
    Unified ResNet model with optional memristive mode
    
    Args:
        block: Type of residual block (BasicBlock/Bottleneck)
        layers: Number of blocks in each layer
        num_classes: Number of output classes
        mem_enabled: Enable memristive layers
        mem_args: Dictionary containing memristive parameters
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        mem_enabled: bool = False,
        mem_args: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.mem_enabled = mem_enabled
        self.mem_args = mem_args if self.mem_enabled else {}
        self.in_channels = 64

        conv_layer = Conv2dMem if mem_enabled else nn.Conv2d
        self.conv1 = conv_layer(
            in_channels=3, out_channels=64, kernel_size=7,
            stride=2, padding=3, bias=False,
            **self.mem_args
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        linear_layer = LinearMem if mem_enabled else nn.Linear
        self.fc = linear_layer(
            in_features=512 * block.expansion, out_features=num_classes,
            **self.mem_args
        )

        # Weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, Conv2dMem)):
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
        if stride != 1 or self.in_channels != channels * block.expansion:
            conv_layer = Conv2dMem if self.mem_enabled else nn.Conv2d
            downsample = nn.Sequential(
                conv_layer(
                    in_channels=self.in_channels, out_channels=channels * block.expansion,
                    kernel_size=1, stride=stride, bias=False,
                    **self.mem_args
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
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def update_weight(self) -> None:
        """Update weights for memristive layers (if enabled)"""
        if not self.mem_enabled:
            return

        for module in self.modules():
            if isinstance(module, (Conv2dMem, LinearMem)):
                module.update_weight()

def ResNet_zoo(
    model_name: str = 'resnet18',
    num_classes: int = 1000,
    pretrained: bool = False,
    mem_enabled: bool = False,
    engine: Optional[Any] = None,
    input_slice: Optional[Union[torch.Tensor, list]] = [1, 1, 2, 4],
    weight_slice: Optional[Union[torch.Tensor, list]] = [1, 1, 2, 4],
    device: Optional[Any] = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    bw_e: Optional[Any] = None
) -> ResNet:
    """
    ResNet model factory
    
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

    """
    mem_args = {
        "engine": engine,
        "input_slice": input_slice,
        "weight_slice": weight_slice,
        "device": device,
        "bw_e": bw_e
    }
    # Architecture configuration
    model_params: Dict[str, Any] = {
        'resnet18': (BasicBlock, [2, 2, 2, 2]),
        'resnet34': (BasicBlock, [3, 4, 6, 3]),
        'resnet50': (Bottleneck, [3, 4, 6, 3]),
        'resnet101': (Bottleneck, [3, 4, 23, 3]),
        'resnet152': (Bottleneck, [3, 8, 36, 3])
    }

    if model_name not in model_params:
        raise ValueError(f"Invalid model name: {model_name}")

    block, layers = model_params[model_name]
    model = ResNet(
        block=block,
        layers=layers,
        num_classes=num_classes,
        mem_enabled=mem_enabled,
        mem_args=mem_args
    )

    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls[model_name]))

    return model
