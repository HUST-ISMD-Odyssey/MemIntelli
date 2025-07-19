# -*- coding:utf-8 -*-
# @File  : linear.py
# @Author: Zhou
# @Date  : 2023/3/22
import os
import sys

import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from memintelli.pimpy.data_formats import SlicedData
from memintelli.NN_layers.functions import linear_mem_func
from matplotlib import pyplot as plt
from memintelli.pimpy import DPETensor
from memintelli.pimpy.utils import SNR

class LinearMem(nn.Module):
    def __init__(self, engine, in_features: int, out_features: int, input_slice:[list, tuple], weight_slice:[list, tuple],
                 bias: bool = True, device=None, dtype=torch.float32, bw_e=None, input_paral_size=(1, 32), weight_paral_size=(32, 32), 
                 input_quant_gran=(1, 32), weight_quant_gran=(32, 32)):
        '''
        :param in_features: the input neuron number
        :param out_features: the output neuron number
        :param bias: use bias or not, default is True
        :param input_sli_mod: the slice method of the input matrix, default is (1, 1, 2, 4)
        :param weight_sli_mod: the slice method of the weight matrix, default is (1, 1, 2, 4)
        :param bw_e: the bit width of the input and weight, default is None, which means use the INT
        :param device: use cuda or cpu, default is None, which means use cpu
        :param dtype:
        '''
        super(LinearMem, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.weight_slice_method = torch.tensor(weight_slice).to(device)
        self.input_slice_method = torch.tensor(input_slice).to(device)

        self.weight_sliced = SlicedData(self.weight_slice_method, device=device, bw_e=bw_e, is_weight=True, paral_size=weight_paral_size, quant_gran=weight_quant_gran)
        self.engine = engine
        self.weight_sliced.slice_data_imp(engine, self.weight.detach().t()) 
        self.input_paral_size = input_paral_size
        self.input_quant_gran = input_quant_gran

    def reset_parameters(self) -> None:
        # Setting x=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_sliced = SlicedData(self.input_slice_method, device=input.device, bw_e=self.weight_sliced.bw_e,is_weight=False, paral_size=self.input_paral_size, quant_gran=self.input_quant_gran)
        input_sliced.slice_data_imp(self.engine, input.detach())
        return linear_mem_func(self.engine, input, self.weight, input_sliced, self.weight_sliced, self.bias)

    def update_weight(self):
        self.weight_sliced.slice_data_imp(self.engine, self.weight.detach().t().to(self.engine.device))

def _test(mode=0):
    if mode == 0:
        print("-----------------cuda-----------------")
        engine = DPETensor(
        HGS=1e-5,                       # High conductance state
        LGS=1e-8,                       # Low conductance state
        write_variation=0.0,          # Write variation
        rate_stuck_HGS=0.001,          # Rate of stuck at HGS
        rate_stuck_LGS=0.000,          # Rate of stuck at LGS
        read_variation={0:0.05, 1:0.05, 2:0.05, 3:0.05},           # Read variation
        vnoise=0.05,                   # Random Gaussian noise of voltage
        rdac=2**2,                      # Number of DAC resolution 
        g_level=2**2,                   # Number of conductance levels
        radc=2**12
        )
        xblk = [1, 1, 2, 2]
        mblk = [1, 1, 2, 2]

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        input = torch.randn(500, 100, requires_grad=True).to(device)
        layer = LinearMem(engine, 100, 300, bias=False, input_slice=xblk, weight_slice=mblk, device=device, bw_e=None, input_paral_size=(1, 32), weight_paral_size=(32, 32), 
                 input_quant_gran=(1, 32), weight_quant_gran=(32, 32))
        output = layer(input)
        #output.backward(torch.ones_like(output, dtype=torch.float))
        weight = layer.weight.data
        #weight.requires_grad = True
        output_ideal = F.linear(input.to(device), weight)
        output_ideal.backward(torch.ones_like(output_ideal, dtype=torch.float))

        output = output.cpu().detach().numpy()
        output_ideal = output_ideal.cpu().detach().numpy()
        print(SNR(output_ideal, output))


if __name__ == '__main__':
    _test(0)