# -*- coding:utf-8 -*-
# @File  : distance.py
# @Author: Zhou
# @Date  : 2024/4/1

import torch.nn as nn
import torch.nn.functional as F
import torch
from MemIntelli.utils import SlicedData, SliceMethod
# from .functions import map_reduce_dot_func
from MemIntelli.NN_layers.functions import map_reduce_dot_func

class EuclideanDistance(nn.Module):
    def __init__(self, vectors, split_square_len, input_sli_med:SliceMethod, weight_sli_med:SliceMethod,
                 device=None):
        '''
        to calculate the Euclidean distance, the formula is:
        sum((x-y)^2) = x^2 + y^2 - 2xy, where x and y are two vectors
        here, y and y^2 is stored in the memory, and x is the input tensor, x^2 is ignored
        sum((x-y)^2) ~= y^2 - 2xy = -1/2 * ((y^2/split_square_len) * split_square_len - xy)
        assume y vectors are weights and (y^2/split_square_len) is the bias

        :param vectors: the vectors to be stored, one column is a vector
        :param split_square_len: the length that used
        :param device: the device that the tensor is stored
        '''
        super(EuclideanDistance, self).__init__()
        self.weight = nn.Parameter(vectors, requires_grad=False)
        self.split_square_len = split_square_len
        self.bias = torch.sum(self.weight ** 2, dim=0) / split_square_len
        temp = torch.cat([self.weight, self.bias], dim=0).to(device)
        self.weight_sliced = SlicedData(temp, weight_sli_med.slice_method, weight_sli_med.bw_e, device=device)
        self.input_slice_method = input_sli_med

    def forward(self, engine, x):
        return euclidean_distance_mem_func(engine, x, self.input_slice_method, self.weight, self.weight_sliced)

    def update_weight(self, weight):
        self.weight = weight
        self.bias = torch.sum(self.weight ** 2, dim=0) / self.split_square_len
        temp = torch.cat([self.weight, self.bias], dim=0)
        self.weight_sliced.update_slice_data(temp)

class EuclideanDistanceMem(torch.autograd.Function):
    @staticmethod
    def forward(ctx, engine, input, input_slice_method, weight, weight_slice):
        ctx.save_for_backward(input.data, weight.data)
        # the transpose of the weight is used to match the F.linear
        if input.shape[1] != weight.shape[0]:
            raise ValueError('The input and weight shape is not matched')
        append = torch.ones((input.shape[0], weight_slice.shape[1] - input.shape[1]), device=input.device)
        input = torch.cat([input, append], dim=-1)
        output = engine.MapReduceDot(input, input_slice_method, weight_slice.t())
        return output

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError('The backward is not implemented')

def euclidean_distance_mem_func(engine, input, input_slice_method, weight, weight_slice):
    return EuclideanDistanceMem.apply(engine, input, input_slice_method, weight, weight_slice)

class CosineDistance(nn.Module):
    '''
    Examples:
        # length is 10
        a = torch.randn(5, 10)
        b = torch.randm(10, 10)
        in_slice = SliceMethod([1,2,4])
        weight_slice = SliceMethod([1,2,4])
        cos_layer = CosineDistance(b, in_slice, weight_slice)
        dist = cos_layer(a)
        # dist.shape is (5*10)
    '''
    def __init__(self, vectors, input_sli_med:SliceMethod, weight_sli_med:SliceMethod,
                 device=None):
        super(CosineDistance, self).__init__()
        # input vector length is the same as the weight vector
        # not support the grad now
        self.weight = nn.Parameter(vectors.t, requires_grad=False).to(device)
        self.weight.data = F.normalize(self.weight, p=2, dim=1)
        self.weight_sliced = SlicedData(self.weight, weight_sli_med.slice_method, weight_sli_med.bw_e, device=device)
        self.input_slice_method = input_sli_med

    def forward(self, engine, x):
        x = F.normalize(x, p=2, dim=1)
        self.weight_sliced.slice_data_imp(engine, self.weight.data)
        return  torch.tensor(1, device=x.device) - map_reduce_dot_func(engine, x, self.input_slice_method,
                                                                       self.weight_sliced)

    def update_weight(self, weight):
        self.weight.data = F.normalize(weight.t, p=2, dim=1)
        self.weight_sliced.update_slice_data(self.weight)

class HammingDistance(nn.Module):
    def __init__(self, vectors, device=None):
        super(HammingDistance, self).__init__()
        self.weight = nn.Parameter(vectors, requires_grad=False)
        self.weight_sliced = SlicedData(self.weight, [1,2], device=device)
        self.input_slice_method = SliceMethod([1,2])

    def forward(self, engine, x):
        return engine.MapReduceDot(x, self.input_slice_method, self.weight_sliced.t)

if __name__ == '__main__':
    from MemIntelli.pimpy import DPETensor
    from MemIntelli.utils import SliceMethod
    import torch

    input = torch.randn(5, 10, requires_grad=True, dtype=torch.float32)
    engine = DPETensor(var=0.05)
    lgth_fraction = 25
    # xblk = [1 for i in range(lgth_fraction - 2)]
    # mblk = [1 for i in range(lgth_fraction - 2)]
    xblk = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 4, 4, 4]
    mblk = [1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 4, 4, 4]
    # xblk = [1, 4, 4]
    # mblk = [1, 4, 4]
    x_slice_method = SliceMethod(xblk, 8)
    m_slice_method = SliceMethod(mblk, 8)

    weight = torch.randn(6, 10, requires_grad=True, dtype=torch.float32)
    cos_layer = CosineDistance(weight, x_slice_method, m_slice_method)
    cos_dist = cos_layer(engine, input)
    print(cos_dist)

    input1 = F.normalize(input, p=2, dim=1)
    weight = F.normalize(weight, p=2, dim=1)
    cos_dist_real = 1-input1.mm(weight.t())
    print(cos_dist_real)
