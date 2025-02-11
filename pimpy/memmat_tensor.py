# -*- coding:utf-8 -*-
# @File  : memmat_tensor.py
# @Author: Zhou
# @Date  : 2024/6/27

'''
this is a new version of the memmat_tensor.py
we use the tensor to realize the dot product, and only consider the INT format data
this version is more efficient than the previous version
'''


import torch
from matplotlib import pyplot as plt
import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.functions import quant_map_tensor,  bfp_map_tensor, SNR
from utils.data_formats import SlicedData

#from MemIntelli.utils import SlicedData, quant_map_tensor, ABSE, bfp_map_tensor, RE,MSE, SNR
import math
import time

def dot_2d(x, y):
    """
    use einsum to calculate the cross 2D product
    :param x: tensor with shape (batch,num_divide_row_a,num_divide, slice_a ,m, n) or (num_divide_row_a,num_divide, slice_a ,m, n)
    :param y: tensor with shape (num_divide,num_divide_col_b, slice_b ,n, p)
    """
    if len(x.shape) == 6:
        return torch.einsum("bnmijk, mpskl->bnmpisjl", x, y)
    elif len(x.shape) == 5:         #batch?
        return torch.einsum("nmijk, mpskl->nmpisjl", x, y)
    else:
        raise ValueError('The input data dimension is not supported!')

class DPETensor(object):
    '''
    use the bit slice method to realize PDE using tensor
    realize the INT format data
    :quant_array_gran:  Quantization array granularity        
        "per-matrix "->per-matrix(the same as matrix size); others e.g. quant_array_gran (128,128)  paral_array_size (64,64),  quant_array_gran needs to be divisible by paral_array_size.
    :quant_input_gran:  Quantization input granularity, the same as quant_array_gran, but for input
    :paral_array_size:  Array size for parallel VMM Computing
    :paral_input_size:  Input size for parallel VMM Computing
    '''
    def __init__(
            self, HGS=1e-4, LGS=1e-8, g_level=2**4, var=0.05, vnoise=0.05, wire_resistance=2.93,
            rdac=2**4, radc=2**12, vread=0.1, quant_array_gran="per-matrix",quant_input_gran="per-matrix",paral_array_size=(64, 64),paral_input_size=(64, 64),device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.HGS = HGS
        self.LGS = LGS
        self.g_level = g_level
        self.var = var
        self.vnoise = vnoise
        self.wire_resistance = wire_resistance
        self.rdac = rdac
        self.radc = radc
        self.vread = vread
        self.quant_array_gran = quant_array_gran
        self.quant_input_gran = quant_input_gran
        self.paral_array_size = paral_array_size
        self.paral_input_size = paral_input_size

        if self.radc < 2:
            raise ValueError('The resolution of the ADC should be larger than 1!')
        if self.rdac < 2:
            raise ValueError('The resolution of the DAC should be larger than 1!')
        if self.g_level < 2:
            raise ValueError('The number of the conductance levels should be larger than 1!')
        if self.LGS >= self.HGS:
            raise ValueError('The low conductance state should be smaller than the high conductance state!')


    def __call__(self, x:SlicedData, mat:SlicedData, wire_factor=False):
        return self.MapReduceDot(x, mat, wire_factor)

    def MapReduceDot(self, x:SlicedData, mat:SlicedData, wire_factor=False):
        '''
        use the MapReduce method to realize the dot product
        :param x: the input tensor with shape (slice, m, n)
        :param x_slice_method: the slice method of the input tensor
        :param mat: the weight tensor with shape (slice, m, p)
        :param wire_factor: whether consider the wire resistance
        :return: the output tensor with shape (m, p)
        '''        
        if mat.device.type != x.device.type:
            raise ValueError('The input data and weight data should be in the same device!')
        if x.quantized_data.shape[-1] != mat.quantized_data.shape[-2]:
            raise ValueError('The input data mismatches the shape of weight data!')
        if wire_factor:
            raise NotImplementedError('The wire_factor is not supported in the tensor version!')
        else: 
            mat.sliced_data = mat.sliced_data.to(self.device)
            mat.sliced_max_weights = mat.sliced_max_weights.to(self.device)
            result = self._dot(x, mat)
        return result

    def _num2R(self, mat:SlicedData):
        # convert the weight data to the resistance and add the noise
        # input dimension (num_divide,num_divide_col_b, slice_b ,n, p)
        # output dimension (num_divide,num_divide_col_b, slice_b ,n, p)
        Q_G = (self.HGS - self.LGS) / (self.g_level - 1)
        max_weights = mat.sliced_max_weights.reshape(1, 1, -1, 1, 1).to(self.device)
        mat.sliced_data = mat.sliced_data.to(self.device)
        G = torch.round(mat.sliced_data / max_weights * (self.g_level - 1)) * Q_G + self.LGS
        r = torch.exp(torch.normal(0, self.var, G.shape, device=self.device))
        return G * r

    def _num2V(self, x:SlicedData):
        # convert input data to the voltage (vread)
        xmax = x.sliced_max_weights
        #without batch, the shape is (num_divide_row_a,num_divide, slice_a ,m, n) or (num_divide_row_a,num_divide, slice_a ,dbfp_slice,m, n)
        if len(x.shape) == 2:       
            if(x.sliced_data.dim()==5):
                xmax = xmax.reshape(1, 1, -1, 1, 1)
            elif(x.sliced_data.dim()==6):
                xmax = xmax.reshape(1, 1, -1, 1, 1, 1)
        #with batch, the shape is (batch,num_divide_row_a,num_divide, slice_a ,m, n) or (batch,num_divide_row_a,num_divide, slice_a ,dbfp_slice,m, n)
        elif len(x.shape) == 3:     
            if(x.sliced_data.dim()==6):
                xmax = xmax.reshape(1, 1, 1, -1, 1, 1)
            elif(x.sliced_data.dim()==7):
                xmax = xmax.reshape(1, 1, 1, -1, 1, 1, 1)
            #print(x.sliced_data.shape)
        else:
            raise ValueError('The input data dimension is not supported!')
        V_in = self.vread * torch.round(x.sliced_data / xmax * (self.rdac - 1)) / (self.rdac - 1)
        return V_in

    def _dot(self, x:SlicedData, mat:SlicedData):
        '''
        calculate the dot product of x and m
        :param x: the input tensor with shape (slice, m, n)
        :param m: the weight tensor with shape (slice, n, p)
        :return: the output tensor with shape (m, p)
        '''
        G = self._num2R(mat)
        Vin = self._num2V(x)
        x.sliced_max_weights = x.sliced_max_weights.to(self.device)
        mat.sliced_max_weights = mat.sliced_max_weights.to(self.device)

        if len(x.shape) == 2:
            adcRef = (self.HGS - self.LGS) * self.vread * Vin.shape[-1]
            QG = (self.HGS - self.LGS) / (self.g_level - 1)
            I = dot_2d(Vin, G - self.LGS)
            I = torch.round(I / adcRef * (self.radc - 1)) / (self.radc - 1)
            temp = torch.mul(I, x.sliced_max_weights.reshape(1, 1, 1, -1, 1, 1, 1)).to(self.device)
            temp = torch.round(torch.mul(temp, mat.sliced_max_weights.to(temp.device).reshape(1, 1, 1, 1, -1, 1, 1)) / QG / self.vread / (self.g_level - 1) * adcRef)
            shift_weights = torch.zeros((len(x),len(mat)), device=x.device)
            
            for i in range(len(x)):
                shift_weights[i] = x.sliced_weights.to(self.device)[i] * mat.sliced_weights.to(self.device)
            out = torch.mul(temp.reshape(temp.shape[0],temp.shape[1],temp.shape[2], -1, temp.shape[5], temp.shape[6]),
                            shift_weights.reshape(1, 1, 1, -1, 1, 1))
            out = out.sum(dim=3) 
            if x.bw_e is None:
            #每一块小矩阵的max_data的乘积
                out_block_max = torch.einsum("nmij, mpij->nmpij",x.max_data.to(self.device) , mat.max_data.to(self.device))
                out = (out* out_block_max
                        / (2 ** (sum(x.slice_method.to(self.device)) - 1) - 1) / (2 ** (sum(mat.slice_method.to(self.device)) - 1) - 1))
            else:
                out_block_e_bias = torch.einsum("nmij, mpij->nmpij",2.**x.e_bias.to(self.device) , 2.**mat.e_bias.to(self.device))
                out = out* out_block_e_bias*2.**(4-sum(x.slice_method)-sum(mat.slice_method))
            out = out.sum(dim=1)
            out = out.permute(0, 2, 1, 3)
            out = out.reshape(out.shape[0] * out.shape[1], out.shape[2] * out.shape[3])
            result = out[:x.shape[0],:mat.shape[1]]
            
        elif len(x.shape) == 3:     
            adcRef = (self.HGS - self.LGS) * self.vread * Vin.shape[-1]
            QG = (self.HGS - self.LGS) / (self.g_level - 1)
            I = dot_2d(Vin, G - self.LGS)
            I = torch.round(I / adcRef * (self.radc - 1)) / (self.radc - 1)
            temp = torch.mul(I, x.sliced_max_weights.reshape(1, 1, 1, 1, -1, 1, 1, 1))
            temp = torch.round(torch.mul(temp, mat.sliced_max_weights.reshape(1, 1, 1, 1, 1, -1, 1, 1))/ QG / self.vread / (self.g_level - 1) * adcRef)
            shift_weights = torch.zeros((len(x),len(mat)), device=x.device)
            
            for i in range(len(x)):
                shift_weights[i] = x.sliced_weights[i] * mat.sliced_weights
            # add the shift weights to the calculated result

            out = torch.mul(temp.reshape(temp.shape[0],temp.shape[1],temp.shape[2],temp.shape[3], -1, temp.shape[6], temp.shape[7]),
                            shift_weights.reshape(1, 1, 1, 1, -1, 1, 1))
            out = out.sum(dim=4) 
            if x.bw_e is None:
            #每一块小矩阵的max_data的乘积
                out_block_max = torch.einsum("bnmij, mpij->bnmpij",x.max_data , mat.max_data)
                out = (out* out_block_max
                        / (2 ** (sum(x.slice_method) - 1) - 1) / (2 ** (sum(mat.slice_method) - 1) - 1))
            else:
                out_block_e_bias = torch.einsum("bnmij, mpij->bnmpij",2.**x.e_bias , 2.**mat.e_bias)
                out = out* out_block_e_bias*2.**(4-sum(x.slice_method)-sum(mat.slice_method))
            out = out.sum(dim=2)
            out = out.permute(0, 1, 3, 2, 4)
            out = out.reshape(out.shape[0],out.shape[1] * out.shape[2], out.shape[3] * out.shape[4])
            #print(x.shape,mat.shape,out.shape)
            result = out[:out.shape[0],:x.shape[1],:mat.shape[1]]
        
        else:
            raise ValueError('The input data dimension is not supported!')

        return result

    def slice_data(self, mat, slice_method, bw_e=None,input_en=False):
        """
        slice the data using the slice method
        :param mat: the data to be sliced, 3D tensor, the shape is (batch, row, col)
        :param slice_method: the slice method, tensor or list
        :param transpose: if transpose is True, then mat should be transpoed before slicing
                            if transpose is False, then mat should be sliced directly
        :param bw_e: the width of the exponent, if bw_e is None, then the data is INT format
        :param input_en: if input_en is True, then the input data and weight data has different size
        :param dbfp_en: if dbfp_en is True, then the input data is dbfp format
        :return:
                data_int: the sliced data in INT format, the shape is (batch, divide_num, slice, row, col)
                mat_data: the data quantized by the slice method, the shape is the same as the input data
                max_mat: the max value of the input data for each slice, the shape is (batch, divide_num, 1, 1, 1)
                e_bias: the bias of the exponent, the shape is (batch, divide_num, slice)
        """
        # take all the input as 4D tensor
        unsqueezed = False

        if len(mat.shape) == 2:
            mat = mat.unsqueeze(0)
            unsqueezed = True

        quant_gran = self.quant_input_gran if input_en else self.quant_array_gran
        paral_size = self.paral_input_size if input_en else self.paral_array_size
        mat = mat.to(self.device)
        if quant_gran == "per-matrix":
            quant_gran = mat.shape[1:]
        elif quant_gran == "per-row":
            quant_gran = (1, mat.shape[2])
        elif quant_gran == "per-col":
            quant_gran = (mat.shape[1], 1)
        else:
            quant_gran = quant_gran
        
        quant_gran = list(quant_gran) 
        #将quant_gran变为paral_size的整数倍
        quant_gran[0] = math.ceil(quant_gran[0] / paral_size[0]) * paral_size[0]
        quant_gran[1] = math.ceil(quant_gran[1] / paral_size[1]) * paral_size[1]

        num_gran_row = math.ceil(mat.shape[1] / quant_gran[0]) 
        num_gran_col = math.ceil(mat.shape[2] / quant_gran[1])
        
        num_divide_row = quant_gran[0] // paral_size[0]
        num_divide_col = quant_gran[1] // paral_size[1]

        temp_mat = torch.zeros((mat.shape[0], num_gran_row * quant_gran[0], num_gran_col * quant_gran[1]), device=mat.device)
        temp_mat[:, :mat.shape[1], :mat.shape[2]] = mat
        temp_mat = temp_mat.reshape(mat.shape[0], num_gran_row, quant_gran[0], num_gran_col, quant_gran[1]).transpose(2, 3)
        max_abs_temp_mat = torch.max(torch.max(torch.abs(temp_mat), dim=-1, keepdim=True)[0], dim=-2, keepdim=True)[0].to(mat.device)
        #max_abs_temp_mat从(mat.shape[0], num_gran_row, num_gran_col,  1,  1)广播到 (mat.shape[0], num_gran_row, num_gran_col, num_divide_row,num_divide_col, 1,  1)
        max_abs_temp_mat = max_abs_temp_mat.unsqueeze(3).unsqueeze(4).expand(-1, -1, -1, num_divide_row, num_divide_col, -1, -1).to(mat.device)
        max_abs_temp_mat = max_abs_temp_mat.transpose(2, 3).reshape(mat.shape[0], num_gran_row * num_divide_row, num_gran_col * num_divide_col, 1, 1).to(mat.device)
        
        temp_mat = temp_mat.reshape(mat.shape[0], num_gran_row, num_gran_col, num_divide_row, paral_size[0], num_divide_col, paral_size[1]).transpose(4, 5)
        temp_mat = temp_mat.transpose(2, 3)
        temp_mat = temp_mat.reshape(mat.shape[0], num_gran_row * num_divide_row, num_gran_col * num_divide_col, paral_size[0], paral_size[1]).to(self.device)
        
        slice_method = slice_method.to(mat.device)
        if bw_e:    # define the FP_map_tensor function
            data_int, mat_data, max_mat, e_bias = bfp_map_tensor(temp_mat, slice_method, bw_e, max_abs_temp_mat)
        else:
            data_int, mat_data, max_mat, e_bias = quant_map_tensor(temp_mat, slice_method, max_abs_temp_mat)
        # the transpose is used to make the data_int is the same as the input data
        mat_data = mat_data.to(self.device)
        max_mat = max_mat.to(self.device)
        e_bias = e_bias.to(self.device) if e_bias is not None else None
        mat_data = mat_data.transpose(2,3).reshape(mat.shape[0],num_gran_row * num_divide_row * paral_size[0],num_gran_col * num_divide_col * paral_size[1])[:,:mat.shape[1],:mat.shape[2]].to(self.device)
            
        # remove the unsqueezed dimension
        if unsqueezed:
            data_int = data_int.squeeze(0)
            mat_data = mat_data.squeeze(0)
            max_mat = max_mat.squeeze(0)
            if e_bias is not None:
                e_bias = e_bias.squeeze(0)
        
        return data_int, mat_data, max_mat, e_bias

if __name__ == '__main__':
    tb_mode = 0
    device = torch.device('cuda:0')
    if tb_mode == 0:
        torch.manual_seed(42)
        x_data = torch.randn(1000, 1000,dtype=torch.float64,device=device)
        mat_data = torch.randn(1000,1000,dtype=torch.float64,device=device)
        mblk = torch.tensor([1,1,2,4])
        xblk = torch.tensor([1,1,2,4])
        mat = SlicedData(mblk, device=device,bw_e=8)
        x = SlicedData(xblk, device=device,bw_e=8,input_en=True)
        size = 64
        paral_size = size
        engine = DPETensor(var=0.00,g_level=16,rdac=16,radc=2**16,quant_array_gran=(size,size),quant_input_gran=(1,size),paral_array_size=(paral_size,paral_size),paral_input_size=(1,paral_size))
        mat.slice_data_imp(engine, mat_data)
        x.slice_data_imp(engine, x_data)
        start = time.time()
        result = engine(x, mat).cpu().numpy()
        rel_result = torch.matmul(x_data, mat_data).cpu().numpy()
        snr_varlue = SNR(result, rel_result)
        print("SNR(dB)",snr_varlue)

        plt.scatter(rel_result.reshape(-1), result.reshape(-1))
        plt.xlabel('Expected Value of Dot Product')
        plt.ylabel('Measured Value of Dot Product')
        plt.show()

        



