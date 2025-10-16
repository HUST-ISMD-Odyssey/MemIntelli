# -*- coding:utf-8 -*-
# @File  : ResNet.py
# @Author: Zhou
# @Date  : 2024/4/1

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, List, Dict, Any, cast, Optional, Type
from torch.hub import load_state_dict_from_url
from memintelli.NN_layers import Conv2dMem, LinearMem

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
        "weight_quant_gran": weight_quant_gran
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
