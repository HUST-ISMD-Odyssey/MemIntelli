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
from memintelli.NN_layers.functions import conv1d_mem_func, conv2d_mem_func
from memintelli.pimpy import DPETensor

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
    def __init__(self, engine, in_channels, out_channels, kernel_size, input_slice:[list, tuple, torch.Tensor],
                 weight_slice:[list, tuple], stride=1, padding=0, dilation=1,bias=True, device=None, dtype=None,
                  bw_e=None, input_paral_size=(1, 32), weight_paral_size=(32, 32), 
                  input_quant_gran=(1, 32), weight_quant_gran=(32, 32)):
        super(Conv2dMem, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Conv2dMem, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        factory_kwargs = {'device': device, 'dtype': dtype}

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
        self.weight_sliced = SlicedData(self.weight_slice_method, device=device,
                                        bw_e=bw_e, is_weight=True, paral_size=weight_paral_size, quant_gran=weight_quant_gran)
        self.engine = engine
        # the sliced weight shape is (C_in*kh*kw, C_out)
        self.weight_sliced.slice_data_imp(engine, self.weight.reshape(self.weight.shape[0], -1).detach().t())

    def reset_parameters(self) -> None:
        # Setting x=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # input_unfold size: (N, C*kh*kw, L), N is the batch size, C is the channel, kh and kw is the kernel size
        # L is the length of the unfolded vector, L = H_out * W_out
        # transpose the input_unfold to (N, L, C*kh*kw)
        input_sliced = SlicedData(self.input_slice_method, device=input.device, bw_e=self.weight_sliced.bw_e,
        is_weight=False, paral_size=self.input_paral_size, quant_gran=self.input_quant_gran)
        input_unfold = F.unfold(input, kernel_size=self.weight.shape[2:], stride=self.stride, padding=self.padding,
                                dilation=self.dilation).transpose(1, 2)
        input_sliced.slice_data_imp(self.engine, input_unfold.detach())
        return conv2d_mem_func(self.engine, input, self.weight, input_sliced, self.weight_sliced, self.bias,
                               self.stride, self.padding, self.dilation)

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



