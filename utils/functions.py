# -*- coding: utf-8 -*-
# @Time    : 2022/6/29 15:08
# @Author  : Zhou
# @FileName: functions.py
# @Software: PyCharm
import inspect
import numpy as np
import torch
from prettytable import PrettyTable
#torch.set_default_dtype(torch.float64)
def generate_noise(row, col, **kwargs):
    try:
        mat = np.random.normal(kwargs['mean'], kwargs['std'], (row, col))
        return mat
    except:
        mat = np.random.random((row,col))
        return mat

def show_params_table(params, header=False):
    """
    Args:
        circuit_params: the params of the circuit, 2-D list
        the format of the params follows ['prams', 'values', 'description']
        each sub-list keeps x single param, and the 2-D list keeps all the params.

    Returns:None
    """
    if header:
        table_header = params.pop(0)
    else:
        table_header = ['prams', 'values', 'description']
    table = PrettyTable(table_header)
    for param in params:
        table.add_row(param)
    print(table)

def legal_location(location, limitation):
    try:
        if len(location) == 1:
            return  location < limitation
        else:
            return np.all((location - limitation) < 0)
    except:
        raise ValueError(location)


def is_index(index):
    """
    judge the input index if is legal index
    Args:
        index: the input index

    Returns:
        bool
    """
    # not modified, uses the Built-in index judgment
    if index is None:
        return False
    return True

def retrieve_name(var):
    '''
    utils:
    get back the name of variables
    '''
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [var_name for var_name, var_val in callers_local_vars if var_val is var]

def use_engine(engine):
    """
    x decorator to use the engine
    Args:
        engine: the engine to be used

    Returns:

    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            return func(engine, *args, **kwargs)
        return wrapper
    return decorator

def quant_map_tensor(mat, blk=(1, 1, 2, 4), max_abs_temp_mat = None):
    '''
    convert the data to the quantized data
    :param mat: 5D tensor (num_divide_row_a,num_divide ,m, n) or 6D tensor (batch,num_divide_row_a,num_divide ,m, n)
    :param blk: slice method
    :return:
        data_int: the quantized data, if mat is 4D, the shape is (num_divide_row_a,num_divide, len(blk) ,m, n),
                    if mat is 5D, the shape is (batch, num_divide_row_a,num_divide, len(blk) ,m, n)
        mat_data: the data after quantization, the same shape as mat
        max_mat: the max value of the mat, the shape is (num_divide_row_a,num_divide, 1, 1) or (batch, num_divide_row_a,num_divide, 1, 1)
        e_bias: None, reserved for the block floating point
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

    '''
    shape_len = len(mat.shape)
    shape = list(mat.shape[:shape_len - 2]) + [len(blk)] + list(mat.shape[shape_len - 2:])
    data_int = torch.empty(shape, device=mat.device, dtype=quant_data_type)
    b = 0
    for idx in range(len(blk)):
        if shape_len == 4:
            data_int[:, :, idx, :, :] = ((matq - matq % 2 ** b) % 2 ** (b + blk[-1 - idx])) >> b
        elif shape_len == 5:
            data_int[:, :, :, idx, :, :] = ((matq - matq % 2 ** b) % 2 ** (b + blk[-1 - idx])) >> b
        b += blk[-1 - idx]
    '''

    if len(mat.shape) == 5:
        data_int = torch.empty((mat.shape[0], mat.shape[1], mat.shape[2], len(blk), mat.shape[3], mat.shape[4]), device=mat.device, dtype=quant_data_type)
        b = 0
        for idx in range(len(blk)): 
            data_int[:, :, :, idx, :, :] = ((matq - matq % 2 ** b) % 2 ** (b + blk[-1 - idx])) >> b
            b += blk[-1 - idx]
    '''
    elif len(mat.shape) == 6:
        data_int = torch.empty((mat.shape[0], mat.shape[1], mat.shape[2], mat.shape[3], len(blk), mat.shape[4], mat.shape[5]), device=mat.device, dtype=quant_data_type)
        b = 0
        for idx in range(len(blk)):
            data_int[:, :, :, :, idx, :, :] = ((matq - matq % 2 ** b) % 2 ** (b + blk[-1 - idx])) >> b
            b += blk[-1 - idx]
    '''
    
    return data_int, mat_data, max_mat, e_bias

def bfp_map_tensor(mat, blk=(1, 1, 2, 4), bw_e=8,  max_abs_temp_mat = None):
    '''
    convert the data to the quantized data with block floating point
    :param mat: 5D tensor (batch,num_divide_row_a,num_divide ,m, n)
    :param blk: slice method
    :return:
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
    # e_bias = torch.full_like(max_mat, -2**(bw_e-1)+1)
    # e_bias[torch.where(max_mat > 0)] = torch.floor(torch.log2(max_mat[torch.where(max_mat > 0)]))
    matq = mat / 2.**e_bias
    matq = torch.round(matq*2.**(bits-2))
    clip_up = (2 ** (bits - 1) - 1).to(mat.device)      
    clip_down = (-2 ** (bits - 1)).to(mat.device)
    matq = torch.clip(matq, clip_down, clip_up)  # round&clip，clip到-2^(bits-1)~2^(bits-1)-1
    mat_data = matq * 2. ** (e_bias + 2 - bits)  # 存储的是反量化后的数据
    location = torch.where(matq < 0)
    matq[location] = 2. ** bits + matq[location]
    if len(mat.shape) == 4:
        data_int = torch.empty((mat.shape[0], mat.shape[1], len(blk), mat.shape[2], mat.shape[3]), device=mat.device, dtype=quant_data_type)
        b = 0
        for idx in range(len(blk)): 
            data_int[:, :, idx, :, :] = ((matq - matq % 2 ** b) % 2 ** (b + blk[-1 - idx])) /(2**b)
            b += blk[-1 - idx]
    elif len(mat.shape) == 5:
        data_int = torch.empty((mat.shape[0], mat.shape[1], mat.shape[2], len(blk), mat.shape[3], mat.shape[4]), device=mat.device, dtype=quant_data_type)
        b = 0
        for idx in range(len(blk)):
            data_int[:, :, :, idx, :, :] = ((matq - matq % 2 ** b) % 2 ** (b + blk[-1 - idx])) /(2**b)
            b += blk[-1 - idx]
   
    return data_int, mat_data, max_mat, e_bias

def ABSE(ytest, ypred):
    return np.sum(np.abs((ytest-ypred)/ytest))/(ytest.shape[0] * ytest.shape[1])

def RE(ytest, ypred):
    return np.sqrt(np.sum((ytest-ypred)**2))/np.sqrt(np.sum(ytest**2))

def MSE(ytest, ypred):
    return np.sum((ytest-ypred)**2)/(ytest.shape[0] * ytest.shape[1])

def SNR(ytest, ypred):
    return 10*np.log10(np.sum(ytest**2)/np.sum((ytest-ypred)**2))

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


if __name__ == '__main__':
    mat = generate_noise(20,20, mean=1, std=0.2)
    print(mat)

