# -*- coding: utf-8 -*-
# @Time    : 2022/6/29 15:08
# @Author  : Zhou
# @FileName: functions.py
# @Software: PyCharm

import numpy as np
import torch


def dot_high_dim(x, y):
    """
    Computes the dot product of two sliced high-dimensional tensors using torch.einsum.

    Parameters:
        x (torch.Tensor): First tensor (shape: (num_divide_row_x, num_divide_col_x, num_slice_x, m, n) or
                                                (batch, num_divide_row_x, num_divide_col_x, num_slice_x, m, n)).
        y (torch.Tensor): Second tensor (shape: (num_divide_row_y, num_divide_col_y, num_slice_y, n, p)).

    Returns:
        torch.Tensor: Result of the dot product (shape: (num_divide_row_x, num_divide_col_y, num_slice_x, num_slice_y, m, p)
                                            or (batch, num_divide_row_x, num_divide_col_y, num_slice_x, num_slice_y, m, p)).
    """
    if len(x.shape) == 5:       #if the input data has no batch
        return torch.einsum("nmijk, mpskl->nmpisjl", x, y)
    elif len(x.shape) == 6:     #if the input data has batch
        return torch.einsum("bnmijk, mpskl->bnmpisjl", x, y)
    else:
        raise ValueError('The input data dimension is not supported!')


def ABSE(ytest, ypred):
    return np.sum(np.abs((ytest-ypred)/ytest))/(ytest.shape[0] * ytest.shape[1])


def SNR(ytest, ypred):
    return 10 * np.log10(np.sum(ytest ** 2) / np.sum((ytest - ypred) ** 2))


def RE(ytest, ypred):
    return np.sqrt(np.sum((ytest-ypred)**2))/np.sqrt(np.sum(ytest**2))


def quant_map_tensor(mat, blk, max_abs_temp_mat = None):
    """
    convert the data to the quantized data

    Parameters:
        mat (torch.Tensor): (batch, num_divide_row, num_divide_col, m, n)
        blk (torch.Tensor): slice method
        max_abs_temp_mat (torch.tensor): the max value of the mat

    Returns:
        data_int (torch.Tensor): the quantized data, the shape is (batch, num_divide_row_a, num_divide, num_slice ,m , n)
        mat_data (torch.Tensor): the data quantized by the slice method, the shape is the same as the data
        max_mat (torch.Tensor): the max value of the data for each quantization granularity, the shape is (batch, num_divide_row_a, num_divide, 1, 1)
        e_bias (torch.Tensor): None, reserved for the block floating point (BFP)
    """
    quant_data_type = torch.uint8 if max(blk) <= 8 else torch.int16
    e_bias = None
    assert blk[0] == 1
    bits = sum(blk)
    if max_abs_temp_mat is None:
        max_mat = torch.max(torch.max(torch.abs(mat), dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0].to(mat.device)
    else:
        max_mat = max_abs_temp_mat

    matq = torch.round(mat / max_mat * (2 ** (bits - 1) - 1)).int()
    mat_data = matq / (2 ** (bits - 1) - 1) * max_mat
    location = torch.where(matq < 0)
    matq[location] = 2 ** bits + matq[location]

    data_int = torch.empty((mat.shape[0], mat.shape[1], mat.shape[2], len(blk), mat.shape[3], mat.shape[4]),
                           device=mat.device, dtype=quant_data_type)
    b = 0
    for idx in range(len(blk)):
        data_int[:, :, :, idx, :, :] = ((matq - matq % 2 ** b) % 2 ** (b + blk[-1 - idx])) >> b
        b += blk[-1 - idx]

    return data_int, mat_data, max_mat, e_bias


def bfp_map_tensor(mat, blk, max_abs_temp_mat=None):
    '''
    convert the data to the block floating point (bfp) data

    Parameters:
        mat (torch.tensor): (batch, num_divide_row, num_divide_col, m, n)
        blk (torch.Tensor): slice method
        bw_e (int): the bit width of the exponent
        max_abs_temp_mat (torch.tensor): the max value of the mat

    Returns:
        data_int (torch.Tensor): the quantized data, the shape is (batch, num_divide_row_a, num_divide, num_slice ,m , n)
        mat_data (torch.Tensor): the data quantized by the slice method, the shape is the same as the data
        max_mat (torch.Tensor): the max value of the data for each quantization granularity,
                                the shape is (batch, num_divide_row_a, num_divide, 1, 1)
        e_bias (torch.Tensor): None, reserved for the block floating point (BFP)
    '''
    quant_data_type = torch.uint8 if max(blk) <= 8 else torch.int16
    assert blk[0] == 1
    bits = sum(blk)
    if max_abs_temp_mat is None:
        max_mat = torch.max(torch.max(torch.abs(mat), dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0].to(mat.device)
    else:
        max_mat = max_abs_temp_mat

    e_bias = torch.floor(torch.log2(max_mat + 1e-10))
    matq = mat / 2. ** e_bias
    matq = torch.round(matq * 2. ** (bits - 2))
    clip_up = (2 ** (bits - 1) - 1).to(mat.device)
    clip_down = (-2 ** (bits - 1)).to(mat.device)
    matq = torch.clip(matq, clip_down, clip_up)  # round & clip，clip到-2^(bits-1) ~ 2^(bits-1)-1
    mat_data = matq * 2. ** (e_bias + 2 - bits)  # mat_data is the dequantized data,
                                                 # which is used to calculate the error of quantization
    location = torch.where(matq < 0)
    matq[location] = 2. ** bits + matq[location]

    data_int = torch.empty((mat.shape[0], mat.shape[1], mat.shape[2], len(blk), mat.shape[3], mat.shape[4]),
                           device=mat.device, dtype=quant_data_type)
    b = 0
    for idx in range(len(blk)):
        data_int[:, :, :, idx, :, :] = ((matq - matq % 2 ** b) % 2 ** (b + blk[-1 - idx])) / (2 ** b)
        b += blk[-1 - idx]

    return data_int, mat_data, max_mat, e_bias
