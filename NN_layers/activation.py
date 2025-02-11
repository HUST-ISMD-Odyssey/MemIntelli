# -*- coding:utf-8 -*-
# @File  : activation.py
# @Author: Zhou
# @Date  : 2024/2/23
"""
realize the activation function with quantization
"""

import torch

def relu_q(x:torch.Tensor, bit_width:int):
    """
    quantization relu function
    Args:
        x: input tensor
        bit_width: bit width of the quantization
    Returns:
        quantized tensor
    """
    scale = 2 ** (bit_width - 1)
    x = torch.clamp(x, 0, scale - 1)
    return x

def sigmoid_q(x:torch.Tensor, bit_width:int):
    """
    quantization sigmoid function
    Args:
        x: input tensor
        bit_width: bit width of the quantization
    Returns:
        quantized tensor
    """
    scale = 2 ** (bit_width - 1)
    x = torch.sigmoid(x)
    x = x * scale
    return x

def tanh_q(x:torch.Tensor, bit_width:int):
    """
    quantization tanh function
    Args:
        x: input tensor
        bit_width: bit width of the quantization
    Returns:
        quantized tensor
    """
    scale = 2 ** (bit_width - 1)
    x = torch.tanh(x)
    x = x * scale
    return x

def softmax_q(x:torch.Tensor, bit_width:int):
    """
    quantization softmax function
    Args:
        x: input tensor
        bit_width: bit width of the quantization
    Returns:
        quantized tensor
    """
    scale = 2 ** (bit_width - 1)
    x = torch.softmax(x, dim=1)
    x = x * scale
    return x

