# -*- coding:utf-8 -*-
# @File  : functions.py
# @Author: Zhou
# @Date  : 2024/4/13

import torch
import torch.nn.functional as F
from memintelli.pimpy.data_formats import SlicedData
import time
from matplotlib import pyplot as plt

# build gradient function for map-reduce dot
class MapReduceDot(torch.autograd.Function):
    @staticmethod
    def forward(ctx, engine, input, input_slice_method, weight_sliced):
        ctx.save_for_backward(input.data, weight_sliced.quantized_data)
        output = engine.MapReduceDot(input, input_slice_method, weight_sliced)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, y = ctx.saved_tensors
        grad_input = grad_weight = None
        if ctx.needs_input_grad[1]:
            grad_input = grad_output.mm(y.t())
        if ctx.needs_input_grad[2]:
            grad_weight = grad_output.t().mm(x)
        return None, grad_input, None, grad_weight

def map_reduce_dot_func(engine, input, input_slice_method, weight_sliced):
    return MapReduceDot.apply(engine, input, input_slice_method, weight_sliced)


class LinearMemRunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, engine, input, weight, input_slice:SlicedData, weight_slice:SlicedData, bias):
        # here, the  input and weight is input for the bp process
        ctx.save_for_backward(input_slice.quantized_data, weight_slice.quantized_data, bias)
        # the transpose of the weight is used to match the F.linear
        input = input.to(engine.device)
        weight = weight.to(engine.device)
        input_slice.sliced_data = input_slice.sliced_data.to(engine.device)
        input_slice.quantized_data = input_slice.quantized_data.to(engine.device)
        input_slice.max_data = input_slice.max_data.to(engine.device)
        input_slice.e_bias = input_slice.e_bias.to(engine.device) if input_slice.e_bias is not None else None
        
        weight_slice.sliced_data = weight_slice.sliced_data.to(engine.device)
        weight_slice.quantized_data = weight_slice.quantized_data.to(engine.device)
        weight_slice.max_data = weight_slice.max_data.to(engine.device)
        weight_slice.e_bias = weight_slice.e_bias.to(engine.device) if weight_slice.e_bias is not None else None
        output = engine.MapReduceDot(input_slice, weight_slice)
        if bias is not None:
            output += bias.to(engine.device)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # in the forward, the calculation has considered the dot engine
        # so in the backward, we directly calculate the gradient of the weight and bias
        input_quant, weight_quant, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None
        weight_quant = weight_quant.to(grad_output.device)
        input_quant = input_quant.to(grad_output.device)
        if ctx.needs_input_grad[1]:
            # 确保 input_quant 的设备与 grad_output 一致
            grad_input = grad_output.matmul(weight_quant.T)
        if ctx.needs_input_grad[2]:
            # 确保 weight_quant 的设备与 grad_output 一致
            grad_weight = grad_output.T.matmul(input_quant)
        if bias is not None and ctx.needs_input_grad[5]:
            # 确保 bias 的设备与 grad_output 一致
            bias = bias.to(grad_output.device)
            grad_bias = grad_output.sum(0).squeeze(0)
        # the returned number is the same as the number of the input
        return None, grad_input, grad_weight, None, None, grad_bias

def linear_mem_func(engine, input, weight, input_slice:SlicedData, weight_slice:SlicedData, bias=None):
    return LinearMemRunc.apply(engine, input, weight,input_slice, weight_slice, bias)

class Conv1dMemRunc(torch.autograd.Function):
    def __init__(self):
        super(Conv1dMemRunc, self).__init__()
        self.stride = 1

    @staticmethod
    def forward(ctx, engine, input, weight, bias=None, stride=1):
        pass


    @staticmethod
    def backward(ctx, grad_output):
        pass

def conv1d_mem_func(engine, input, weight, bias=None, stride=1):
    return Conv1dMemRunc.apply(engine, input, weight, bias, stride)

class Conv2dMemRunc(torch.autograd.Function):
    def __init__(self):
        super(Conv2dMemRunc, self).__init__()
        self.stride = 1
        self.padding = 0
        self.dilation = 1

    @staticmethod
    def forward(ctx, engine, input:torch.Tensor, weight:torch.Tensor, input_sliced:SlicedData,
                weight_sliced:SlicedData, bias=None, stride=1, padding=0, dilation=1):
        """
        :param ctx:
        :param engine: dot product engine
        :param input: input feature map, size: (N, C, H, W)
        :param weight: weight, size: (C_out, C_in, kh, kw)
        :param slice_method: data slice method of the weight
        :param bias: bias, size: (C_out)
        :param stride:
        :param padding:
        :param dilation:
        :return: the output feature map, size: (N, C_out, H_out, W_out)
        """
        # change stride, padding, dilation to tuple
        stride, padding, dilation = is_tuple_2(stride), is_tuple_2(padding), is_tuple_2(dilation)
        ctx.stride, ctx.padding, ctx.dilation = stride, padding, dilation

        # H_out = (H_in + 2*padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
        # W_out = (W_in + 2*padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1
        # todo: varify numpy.lib.stride_tricks.as_strided : https://zhuanlan.zhihu.com/p/64933417
        input_shape = input.shape
        ctx.save_for_backward(input_sliced.quantized_data, weight_sliced.quantized_data, bias,
                              torch.tensor(input.shape), torch.tensor(weight.shape))

        h_out = int((input_shape[2] + 2 * padding[0] - dilation[0] * (weight.shape[2] - 1) - 1) / stride[0] + 1)
        w_out = int((input_shape[3] + 2 * padding[1] - dilation[1] * (weight.shape[3] - 1) - 1) / stride[1] + 1)
        out = engine.MapReduceDot(input_sliced, weight_sliced)

        if bias is not None:
            out += bias
        out = F.fold(out.transpose(1, 2), output_size=(h_out, w_out), kernel_size=(1, 1))
        # out = out.transpose(1, 2).reshape(out.shape[0], out.shape[-1], h_out, w_out)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # in the forward, the calculation has considered the dot engine
        # so in the backward, we directly calculate the gradient of the weight and bias
        input_quant, weight_quant, bias, input_shape, weight_shape = ctx.saved_tensors

        # calculate the overlapped block values during unfold-fold.
        ones = torch.ones(torch.Size(input_shape), device=input_quant.device)
        ones = F.unfold(ones, kernel_size=weight_shape[2:], stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
        ones = F.fold(ones, output_size=(input_shape[2], input_shape[3]), kernel_size=weight_shape[2:],
                      stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
        
        ones[torch.where(ones==0)] = 1
        #ones = torch.where(ones == 0, torch.ones_like(ones), ones)

        input_quant = input_quant.transpose(1, 2)
        input_quant = F.fold(input_quant , output_size=(input_shape[2], input_shape[3]), kernel_size=weight_shape[2:],
                            stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation) / ones

        weight_shape_dilated = (weight_shape[0], weight_shape[1], (weight_shape[2] - 1) * ctx.dilation[0] + 1,
                                (weight_shape[3] - 1) * ctx.dilation[1] + 1)

        grad_input = grad_weight = grad_bias = None
        if grad_output is None:
            return None, None, None, None, None, None, None, None, None
        if ctx.stride[0] > 1 or ctx.stride[1] > 1:
            # zero padding
            # [[1,2],           [[1,0,2,0],
            #           -->      [0,0,0,0],
            # [3,4]]             [3,0,4,0]],
            def interleave_index(x, k1, k2):
                *cdims, Hin, Win = x.shape
                Hout = (k1 + 1) * (Hin - 1) + 1
                Wout = (k2 + 1) * (Win - 1) + 1
                out = x.new_zeros(*cdims, Hout, Wout)
                out[..., :: k1 + 1, :: k2 + 1] = x
                return out
            grad_output = interleave_index(grad_output, ctx.stride[0] - 1, ctx.stride[1] - 1)

        # calculate the gradient of input using F.conv2d
        if ctx.needs_input_grad[1]:
            weight_re = weight_quant.T.reshape((weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]))
            # rot weight 180
            weight_re = torch.rot90(weight_re, k=2, dims=[2, 3])
            # exchange the 0 and 1 dimensions of weight
            weight_re = weight_re.transpose(0,1)

            grad_input_pad = F.conv2d(grad_output, weight_re, stride=1, padding=(weight_shape_dilated[2] - 1, weight_shape_dilated[3] - 1), dilation=ctx.dilation)

            clipH = (input_quant.shape[2] - weight_shape_dilated[2] + 2 * ctx.padding[0]) % ctx.stride[0]
            clipW = (input_quant.shape[3] - weight_shape_dilated[3] + 2 * ctx.padding[1]) % ctx.stride[1]

            # Note: the rule of F.pad is (padding_left, padding_right, padding_top, padding_bottom)
            grad_input = F.pad(grad_input_pad,(0, clipW, 0, clipH))

            if ctx.padding[0] != 0 and ctx.padding[1] != 0:
                grad_input = grad_input[..., ctx.padding[0]:-ctx.padding[0], ctx.padding[1]:-ctx.padding[1]]
            else:
                if ctx.padding[0] != 0:
                    grad_input = grad_input[..., ctx.padding[0]:-ctx.padding[0], :]
                if ctx.padding[1] != 0:
                    grad_input = grad_input[..., ctx.padding[1]:-ctx.padding[1]]
            # del grad_output_pad, weight_re

        # calculate the gradient of weight using dot engine
        # weight_quant shape: [Cin*kH*kW, Cout]
        # input_quant shape:  [b, L, Cin*kH*kW]
        # input_shape, weight_shape: [b, Cin, H, W], [Cout, Cin, kH, kW]
        # grad_output shape:  [b, Cout, E, F]
        # if ctx.needs_input_grad[1]:
        #     grad_shape = grad_output.shape  # [b, Cout, E', F']
        #     grad_output_p = grad_output.reshape([grad_shape[0], grad_shape[1], -1]).transpose(1,2)    # [b, E'*F', Cout]

        #     weight_quant_p = weight_quant.transpose(0,1)    # [Cout, Cin*kH*kW]
            
        #     # use torch.matmul
        #     grad_input_p = torch.matmul(grad_output_p, weight_quant_p)    # [b, E'*F', Cin*kH*kW]
        #     grad_input_p = grad_input_p.transpose(1,2).reshape([grad_shape[0], input_shape[1], weight_shape[2], weight_shape[3], grad_shape[2], grad_shape[3]]) # [b, Cin, kH, kW, E', F']

        #     grad_input = torch.zeros(grad_input_p.shape[0], grad_input_p.shape[1], grad_input_p.shape[2], grad_input_p.shape[3], 
        #                              grad_input_p.shape[4]+weight_shape_dilated[2]-1, grad_input_p.shape[5]+weight_shape_dilated[3]-1, device=grad_input_p.device)

        #     for iH in range(weight_shape[2]):
        #         for iW in range(weight_shape[3]):
        #             grad_input[:, :, iH, iW, iH*ctx.dilation[0]:(iH*ctx.dilation[0]+grad_input_p.shape[-2]), iW*ctx.dilation[1]:(iW*ctx.dilation[1]+grad_input_p.shape[-1])] = grad_input_p[:, :, iH, iW, :, :]

        #     grad_input = grad_input.sum(dim=(2,3))

        #     clipH = (input_quant.shape[2] - weight_shape_dilated[2] + 2 * ctx.padding[0]) % ctx.stride[0]
        #     clipW = (input_quant.shape[3] - weight_shape_dilated[3] + 2 * ctx.padding[1]) % ctx.stride[1]

        #     # Note: the rule of F.pad is (padding_left, padding_right, padding_top, padding_bottom)
        #     grad_input = F.pad(grad_input,(0, clipW, 0, clipH))

        #     if ctx.padding[0] != 0 and ctx.padding[1] != 0:
        #         grad_input = grad_input[..., ctx.padding[0]:-ctx.padding[0], ctx.padding[1]:-ctx.padding[1]]
        #     else:
        #         if ctx.padding[0] != 0:
        #             grad_input = grad_input[..., ctx.padding[0]:-ctx.padding[0], :]
        #         if ctx.padding[1] != 0:
        #             grad_input = grad_input[..., ctx.padding[1]:-ctx.padding[1]]

        if ctx.needs_input_grad[2]:
            input = input_quant.transpose(0, 1)

            # Note: the rule of F.pad is (padding_left, padding_right, padding_top, padding_bottom)
            input = F.pad(input,(ctx.padding[1], ctx.padding[1], ctx.padding[0], ctx.padding[0]))

            if ctx.stride[0] != 1:
                row_clip = (input_quant.shape[2] - weight_shape_dilated[2] + 2 * ctx.padding[0]) % ctx.stride[0]
                if row_clip != 0:
                    input = input[..., :-row_clip, :]

            if ctx.stride[1] != 1:
                col_clip = (input_quant.shape[3] - weight_shape_dilated[3] + 2 * ctx.padding[1]) % ctx.stride[1]
                if col_clip != 0:
                    input = input[..., :, :-col_clip]

            grad_weight = F.conv2d(input, grad_output.transpose(0, 1), stride=1, padding=0, dilation=1).transpose(0, 1)
            # inverse_dilation
            grad_weight = grad_weight[..., ::ctx.dilation[0], ::ctx.dilation[1]]

        if bias is not None and ctx.needs_input_grad[5]:
            grad_bias = grad_output.sum(0).sum(-1).sum(-1)

        return None, grad_input, grad_weight, None, None, grad_bias, None, None, None

    '''original backward code save backup'''
    # @staticmethod
    # def backward(ctx, grad_output):
    #     # in the forward, the calculation has considered the dot engine
    #     # so in the backward, we directly calculate the gradient of the weight and bias
    #     input_quant, weight_quant, bias, input_shape, weight_shape = ctx.saved_tensors

    #     ones = torch.ones(torch.Size(input_shape), device=input_quant.device)
    #     ones = F.unfold(ones, kernel_size=weight_shape[2:], stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)
    #     ones = F.fold(ones, output_size=(input_shape[2], input_shape[3]), kernel_size=weight_shape[2:],
    #                   stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation)

    #     input_quant = input_quant.transpose(1, 2)
    #     weight_quant = weight_quant.T.reshape((weight_shape[0], weight_shape[1], weight_shape[2], weight_shape[3]))
    #     input_quant = F.fold(input_quant , output_size=(input_shape[2], input_shape[3]), kernel_size=weight_shape[2:],
    #                         stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation) / ones


    #     grad_input = grad_weight = grad_bias = None
    #     if grad_output is None:
    #         return None, None, None, None, None, None, None, None, None
    #     if ctx.stride[0] > 1:
    #         # zero padding
    #         # [[1,2],           [[1,0,2,0],
    #         #           -->      [0,0,0,0],
    #         # [3,4]]             [3,0,4,0]],
    #         def interleave_index(x, k):
    #             *cdims, Hin, Win = x.shape
    #             Hout = (k + 1) * (Hin - 1) + 1
    #             Wout = (k + 1) * (Win - 1) + 1
    #             out = x.new_zeros(*cdims, Hout, Wout)
    #             out[..., :: k + 1, :: k + 1] = x
    #             return out
    #         grad_output = interleave_index(grad_output, ctx.stride[0] - 1)

    #     if ctx.needs_input_grad[1]:
    #         # rot weight 180
    #         weight_quant = torch.rot90(weight_quant, k=2, dims=[2, 3])
    #         # exchange the 0 and 1 dimensions of weight
    #         weight_re = weight_quant.transpose(0,1)
    #         padding = (weight_quant.shape[2] - 1 - ctx.padding[0], weight_quant.shape[3] - 1 - ctx.padding[1])
    #         clip = (input_quant.shape[2] - weight_quant.shape[2] + 2 * padding[0]) % ctx.stride[0]

    #         # todo:optimize the runtime time
    #         if clip != 0:
    #             padding = (padding[0], padding[0] + clip, padding[1], padding[1] + clip)
    #             grad_output_pad = F.pad(grad_output, padding)
    #         else:
    #             grad_output_pad = grad_output
    #         grad_input = F.conv2d(grad_output_pad, weight_re, stride=1, padding=0 if clip != 0 else padding,
    #                               dilation=ctx.dilation)
    #         # del grad_output_pad, weight_re

    #     if ctx.needs_input_grad[2]:
    #         if ctx.stride[0] == 1:
    #             input = input_quant.mean(0).unsqueeze(1)
    #             # grad_weight = F.conv2d(input, grad_output.mean(0).unsqueeze(1), stride=ctx.stride, padding=ctx.padding,
    #             #                        dilation=ctx.dilation).transpose(0, 1)
    #             grad_weight = F.conv2d(input, grad_output.sum(0).unsqueeze(1), stride=ctx.stride, padding=ctx.padding,
    #                                    dilation=ctx.dilation).transpose(0, 1)
    #         else:
    #             # clip the input feature map
    #             row_clip = (input_quant.shape[2] - weight_quant.shape[2] + 2 * ctx.padding[0]) % ctx.stride[0]
    #             col_clip = (input_quant.shape[3] - weight_quant.shape[3] + 2 * ctx.padding[1]) % ctx.stride[1]
    #             if row_clip != 0 or col_clip != 0:
    #                 input_quant = input_quant[:, :, :-row_clip, :-col_clip]
    #             # padding = ctx.padding
    #             input_quant = input_quant.mean(0).unsqueeze(1)
    #             # grad_weight = F.conv2d(input, grad_output.mean(0).unsqueeze(1), stride=1, padding=ctx.padding,
    #             #                         dilation=ctx.dilation).transpose(0, 1)
    #             grad_weight = F.conv2d(input_quant, grad_output.sum(0).unsqueeze(1), stride=1, padding=ctx.padding,
    #                                    dilation=ctx.dilation).transpose(0, 1)
    #     if bias is not None and ctx.needs_input_grad[5]:
    #         grad_bias = grad_output.sum(-1).sum(-1)
    #     return None, grad_input, grad_weight, None, None, grad_bias, None, None, None

def conv2d_mem_func(engine, input, weight, input_slice:SlicedData, weight_slice:SlicedData, bias=None, stride=1, padding=0, dilation=1):
   return Conv2dMemRunc.apply(engine, input, weight, input_slice, weight_slice, bias, stride, padding, dilation)

def is_tuple_2(x):
    # if x is x tuple of 2 elements, return x, else return (x, x)
    if isinstance(x, tuple):
        assert len(x) == 2
        return x
    elif isinstance(x, int):
        return (x, x)
    else:
        raise ValueError("x must be x tuple or int")
