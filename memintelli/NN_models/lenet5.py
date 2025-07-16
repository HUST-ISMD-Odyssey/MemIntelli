# -*- coding:utf-8 -*-
# @File  : lenet5.py
# @Author: Zhou
# @Date  : 2024/5/8
import torch
import torch.nn as nn
import torch.nn.functional as F
from memintelli.NN_layers import Conv2dMem, LinearMem
from typing import Union, List, Dict, Any, cast, Optional, Type

class LeNet5(nn.Module):
    def __init__(self, 
        mem_enabled: bool = True,
        engine: Optional[Any] = None,
        input_slice: Optional[Union[torch.Tensor, list]] = [1, 1, 2, 4],
        weight_slice: Optional[Union[torch.Tensor, list]] = [1, 1, 2, 4],
        device: Optional[Any] = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        bw_e: Optional[Any] = None
    ):
        super().__init__()
        self.mem_enabled = mem_enabled
        self.mem_args = {
            'engine': engine,
            'input_slice': input_slice,
            'weight_slice': weight_slice,
            'device': device,
            'bw_e': bw_e
        } if mem_enabled else {}
        conv_layer = Conv2dMem if mem_enabled else nn.Conv2d
        linear_layer = LinearMem if mem_enabled else nn.Linear
        self.conv1 = conv_layer(in_channels=1, out_channels=6, kernel_size=5, **self.mem_args)
        self.conv2 = conv_layer(in_channels=6, out_channels=16, kernel_size=5, **self.mem_args)
        self.fc1 = linear_layer(in_features=16*4*4, out_features=120, **self.mem_args)
        self.fc2 = linear_layer(in_features=120, out_features=84, **self.mem_args)
        self.fc3 = linear_layer(in_features=84, out_features=10, **self.mem_args)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16*4*4)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x

    def update_weight(self):
        """Update weights for memory-efficient layers (if enabled)."""
        if not self.mem_enabled:
            return

        for module in self.modules():
            if isinstance(module, (LinearMem, Conv2dMem)):
                module.update_weight()

