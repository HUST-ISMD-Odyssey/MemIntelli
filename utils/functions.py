# -*- coding: utf-8 -*-
# @Time    : 2025/02/20 15:16
# @Author  : Zhou
# @FileName: functions.py
'''
This file is used to quantize the data and convert the data to the INT or block floating point (BFP) data.
Also, it provides some functions to describe the error of the quantization.
'''

import numpy as np
import torch

def quant_map_tensor(mat, blk = (1, 1, 2, 4), max_abs_temp_mat = None):
    '''
    convert the data to the quantized data

    Parameters:
        mat (torch.tensor): (batch, num_divide_row, num_divide_col, m, n)
        blk (tuple): slice method
        max_abs_temp_mat (torch.tensor): the max value of the mat 

    Returns:
        data_int (torch.Tensor): the quantized data, the shape is (batch, num_divide_row_a, num_divide, num_slice ,m , n)
        mat_data (torch.Tensor): the data quantized by the slice method, the shape is the same as the data
        max_mat (torch.Tensor): the max value of the data for each quantization granularity, the shape is (batch, num_divide_row_a, num_divide, 1, 1)
        e_bias (torch.Tensor): None, reserved for the block floating point (BFP)
    '''
    quant_data_type = torch.uint8 if max(blk)<=8 else torch.int16
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

    data_int = torch.empty((mat.shape[0], mat.shape[1], mat.shape[2], len(blk), mat.shape[3], mat.shape[4]), device=mat.device, dtype=quant_data_type)
    b = 0
    for idx in range(len(blk)): 
        data_int[:, :, :, idx, :, :] = ((matq - matq % 2 ** b) % 2 ** (b + blk[-1 - idx])) >> b
        b += blk[-1 - idx]
    
    return data_int, mat_data, max_mat, e_bias

def bfp_map_tensor(mat, blk=(1, 1, 2, 4), bw_e=8,  max_abs_temp_mat = None):
    '''
    convert the data to the block floating point (bfp) data

    Parameters:
        mat (torch.tensor): (batch, num_divide_row, num_divide_col, m, n)
        blk (tuple): slice method
        bw_e (int): the bit width of the exponent
        max_abs_temp_mat (torch.tensor): the max value of the mat 

    Returns:
        data_int (torch.Tensor): the quantized data, the shape is (batch, num_divide_row_a, num_divide, num_slice ,m , n)
        mat_data (torch.Tensor): the data quantized by the slice method, the shape is the same as the data
        max_mat (torch.Tensor): the max value of the data for each quantization granularity, the shape is (batch, num_divide_row_a, num_divide, 1, 1)
        e_bias (torch.Tensor): None, reserved for the block floating point (BFP)
    '''
    quant_data_type = torch.uint8 if max(blk) <= 8 else torch.int16
    assert blk[0] == 1
    bits = sum(blk)
    abs_mat = torch.abs(mat)
    if max_abs_temp_mat is None:
        max_mat = torch.max(torch.max(torch.abs(mat), dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0].to(mat.device)
    else:
        max_mat = max_abs_temp_mat
    
    e_bias = torch.full_like(max_mat, 0)
    e_bias = torch.floor(torch.log2(max_mat+1e-10))
    matq = mat / 2.**e_bias
    matq = torch.round(matq*2.**(bits-2))
    clip_up = (2 ** (bits - 1) - 1).to(mat.device)      
    clip_down = (-2 ** (bits - 1)).to(mat.device)
    matq = torch.clip(matq, clip_down, clip_up)  # round&clip，clip到-2^(bits-1)~2^(bits-1)-1
    mat_data = matq * 2. ** (e_bias + 2 - bits)  # mat_data is the dequantized data, which is used to calculate the error of quantization
    location = torch.where(matq < 0)
    matq[location] = 2. ** bits + matq[location]

    data_int = torch.empty((mat.shape[0], mat.shape[1], mat.shape[2], len(blk), mat.shape[3], mat.shape[4]), device=mat.device, dtype=quant_data_type)
    b = 0
    for idx in range(len(blk)):
        data_int[:, :, :, idx, :, :] = ((matq - matq % 2 ** b) % 2 ** (b + blk[-1 - idx])) /(2**b)
        b += blk[-1 - idx]
   
    return data_int, mat_data, max_mat, e_bias

'''----------------------------------Some functions to describe the error----------------------------------'''
def ABSE(ytest, ypred):
    return np.sum(np.abs((ytest-ypred)/ytest))/(ytest.shape[0] * ytest.shape[1])

def RE(ytest, ypred):
    return np.sqrt(np.sum((ytest-ypred)**2))/np.sqrt(np.sum(ytest**2))

def MSE(ytest, ypred):
    return np.sum((ytest-ypred)**2)/(ytest.shape[0] * ytest.shape[1])

def SNR(ytest, ypred):
    return 10*np.log10(np.sum(ytest**2)/np.sum((ytest-ypred)**2))

'''----------------------------------Old Version of dec_2FP_map in Pimpy----------------------------------'''
def dec_2FP_map(decmat, blk=(1, 2, 2, 2, 4, 4, 4, 4), bw_e=8):
    newblk = [1, 1] + blk
    num_blk = len(newblk)
    max_a = np.max(np.abs(decmat))
    e_bia = 0
    if max_a >= 2:
        while (max_a >= 2):
            max_a /= 2
            e_bia += 1
    elif (max_a < 1) and (max_a > 0):
        while ((max_a < 1) and (max_a > 0)):
            max_a *= 2
            e_bia -= 1
    else:
        e_bia = 0

    decmat_aliE = decmat / 2 ** e_bia
    decmat_aliE[np.where(decmat_aliE < 0)] = 4 + decmat_aliE[np.where(decmat_aliE < 0)]

    b = np.zeros((num_blk, decmat.shape[0], decmat.shape[1]))
    w = 0
    for i in range(num_blk):
        w = w + newblk[i]
        b[i, :, :] = (decmat_aliE / 2 ** (2 - w)).astype('int')
        decmat_aliE -= b[i, :, :] * (2 ** (2 - w))
    e_max_range = 2 ** (bw_e - 1) - 1

    return np.clip(np.array([e_bia]), -e_max_range, e_max_range), b

def dec_2FP_map_tensor(decmat, blk, bw_e):
    newblk = blk
    num_blk = len(newblk)
    max_a = torch.max(torch.abs(decmat))
    e_bia = 0
    if max_a >= 2:
        while (max_a >= 2):
            max_a /= 2
            e_bia += 1
    elif (max_a < 1) and (max_a > 0):
        while ((max_a < 1) and (max_a > 0)):
            max_a *= 2
            e_bia -= 1
    else:
        e_bia = 0

    decmat_aliE = decmat / 2 ** e_bia
    decmat_aliE[torch.where(decmat_aliE < 0)] = 4 + decmat_aliE[torch.where(decmat_aliE < 0)]

    b = torch.zeros((num_blk, decmat.shape[0], decmat.shape[1]), device=decmat.device)
    w = 0
    for i in range(num_blk):
        w = w + newblk[i]
        b[i, :, :] = (decmat_aliE / 2 ** (2 - w)).int()
        decmat_aliE -= b[i, :, :] * (2 ** (2 - w))
    e_max_range = 2 ** (bw_e - 1) - 1

    return torch.clamp(torch.Tensor([e_bia]), -e_max_range, e_max_range), b



