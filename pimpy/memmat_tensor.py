# -*- coding:utf-8 -*-
# @File  : memmat_tensor.py
# @Author: ZZW
# @Date  : 2025/02/19

'''
this is a new version of the memmat_tensor.py
Optimizes matrix multiplication processes using tensor operations, supporting both integer (INT) and floating-point (FP) data formats or enhanced efficiency and accuracy.
'''
import torch
from matplotlib import pyplot as plt
import sys
import os 
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.functions import quant_map_tensor, bfp_map_tensor, SNR
from utils.data_formats import SlicedData
import math
import time

class DPETensor(object):
    '''
    Implements a dot product engine using bit-sliced tensor operations for matrix multiplication.
    Supports INT and FP data formats with configurable quantization granularity and device settings.

    Parameters:
        HGS (float): High conductance state
        LGS (float): Low conductance state
        g_level (int): Number of conductance levels
        var (float): Random Gaussian noise of conductance
        vnoise (float): Random Gaussian noise of voltage
        wire_resistance (float): Wire resistance
        rdac (int): Number of DAC resolution
        radc (int): Number of ADC resolution
        vread (float): Read voltage
        weight_quant_gran (str or tuple): Quantization granularity of the weight matrix
            "per-matrix" -> The whole matrix is quantized together (i.e., the quantization granularity is (m, n) the same as the matrix shape).
            "per-row" -> Each row of the matrix is quantized separately. (i.e., the quantization granularity is (1, n)).
            "per-col" -> Each column of the matrix is quantized separately. (i.e., the quantization granularity is (m, 1)).
            (a, b) -> The quantization granularity is (a, b).
        input_quant_gran (str or tuple): Quantization granularity of the input matrix
        input_paral_size (tuple): The size of the input matrix used for parallel computation
        weight_paral_size (tuple): The size of the weight matrix used for parallel computation
    '''
    def __init__(
            self, HGS = 1e-4, LGS = 1e-8, g_level = 2**4, var = 0.05, vnoise = 0.05, wire_resistance = 2.93, rdac = 2**4, radc = 2**12, vread = 0.1, 
            weight_quant_gran = "per-matrix", input_quant_gran = "per-matrix", weight_paral_size = (64, 64), input_paral_size = (64, 64), 
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
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
        self.weight_quant_gran = weight_quant_gran
        self.input_quant_gran = input_quant_gran
        self.weight_paral_size = weight_paral_size
        self.input_paral_size = input_paral_size

        if self.radc < 2:
            raise ValueError('The resolution of the ADC should be larger than 1!')
        if self.rdac < 2:
            raise ValueError('The resolution of the DAC should be larger than 1!')
        if self.g_level < 2:
            raise ValueError('The number of the conductance levels should be larger than 1!')
        if self.LGS >= self.HGS:
            raise ValueError('The low conductance state should be smaller than the high conductance state!')


    def __call__(self, x: SlicedData, mat: SlicedData, wire_factor=False):
        return self.MapReduceDot(x, mat, wire_factor)

    def MapReduceDot(self, x: SlicedData, mat: SlicedData, wire_factor = False):
        """
        Implements matrix multiplication using the MapReduce method.

        Parameters:
            x (SlicedData): Input tensor (shape: (m, n) or (batch, m, n)).
            mat (SlicedData): Weight tensor (shape: (n, p)).
            wire_factor (bool): Consider wire resistance (not implemented).

        Returns:
            torch.Tensor: Result of the matrix multiplication.
        """      
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

    def _num2R(self, mat: SlicedData):
        """
        Converts weight data to resistance with added normal noise.

        Parameters:
            mat (SlicedData): Weight data.

        Returns:
            torch.Tensor: Resistance values.
        """
        Q_G = (self.HGS - self.LGS) / (self.g_level - 1)
        max_weights = mat.sliced_max_weights.reshape(1, 1, -1, 1, 1).to(self.device)
        mat.sliced_data = mat.sliced_data.to(self.device)
        G = torch.round(mat.sliced_data / max_weights * (self.g_level - 1)) * Q_G + self.LGS
        r = torch.exp(torch.normal(0, self.var, G.shape, device=self.device))
        return G * r

    def _num2V(self, x: SlicedData):
        """
        Converts input data to voltage (scaled by read voltage).

        Parameters:
            x (SlicedData): Input data.

        Returns:
            torch.Tensor: Voltage values.
        """
        xmax = x.sliced_max_weights
        if len(x.shape) == 2:        #without batch, the shape is (num_divide_row_x, num_divide_col_x, num_slice_x, m, n)
            xmax = xmax.reshape(1, 1, -1, 1, 1)
        elif len(x.shape) == 3:      #with batch, the shape is (batch, num_divide_row_x, num_divide_col_x, num_slice_x, m, n)
            xmax = xmax.reshape(1, 1, 1, -1, 1, 1)
        else:
            raise ValueError('The input data dimension is not supported!')
        V_in = self.vread * torch.round(x.sliced_data / xmax * (self.rdac - 1)) / (self.rdac - 1)
        return V_in

    def _dot(self, x: SlicedData, mat: SlicedData):
        """
        Computes the dot product of input and weight tensors.

        Parameters:
            x (SlicedData): Input tensor with shape (m, n) or (batch, m, n).
            mat (SlicedData): Weight tensor with shape (n, p).

        Returns:
            torch.Tensor: Result of the dot product with shape (m, p) or (batch, m, p).
        """
        G = self._num2R(mat)
        Vin = self._num2V(x)
        x.sliced_max_weights = x.sliced_max_weights.to(self.device)
        mat.sliced_max_weights = mat.sliced_max_weights.to(self.device)

        if len(x.shape) == 2:    #if the input data has no batch
            adcRef = (self.HGS - self.LGS) * self.vread * Vin.shape[-1]
            QG = (self.HGS - self.LGS) / (self.g_level - 1)
            I = dot_high_dim(Vin, G - self.LGS)
            I = torch.round(I / adcRef * (self.radc - 1)) / (self.radc - 1)
            temp = torch.mul(I, x.sliced_max_weights.reshape(1, 1, 1, -1, 1, 1, 1)).to(self.device)
            temp = (torch.mul(temp, mat.sliced_max_weights.to(temp.device).reshape(1, 1, 1, 1, -1, 1, 1)) / QG / self.vread / (self.g_level - 1) * adcRef)
            shift_weights = torch.zeros((len(x),len(mat)), device=x.device)
            
            for i in range(len(x)):
                shift_weights[i] = x.sliced_weights.to(self.device)[i] * mat.sliced_weights.to(self.device)
            out = torch.mul(temp.reshape(temp.shape[0],temp.shape[1],temp.shape[2], -1, temp.shape[5], temp.shape[6]),
                            shift_weights.reshape(1, 1, 1, -1, 1, 1))
            out = out.sum(dim=3) 
            if x.bw_e is None:
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
            I = dot_high_dim(Vin, G - self.LGS)
            I = torch.round(I / adcRef * (self.radc - 1)) / (self.radc - 1)
            temp = torch.mul(I, x.sliced_max_weights.reshape(1, 1, 1, 1, -1, 1, 1, 1))
            temp = (torch.mul(temp, mat.sliced_max_weights.reshape(1, 1, 1, 1, 1, -1, 1, 1))/ QG / self.vread / (self.g_level - 1) * adcRef)
            shift_weights = torch.zeros((len(x),len(mat)), device=x.device)
            
            for i in range(len(x)):
                shift_weights[i] = x.sliced_weights[i] * mat.sliced_weights
            # add the shift weights to the calculated result
            out = torch.mul(temp.reshape(temp.shape[0],temp.shape[1],temp.shape[2],temp.shape[3], -1, temp.shape[6], temp.shape[7]),
                            shift_weights.reshape(1, 1, 1, 1, -1, 1, 1))
            out = out.sum(dim=4) 
            if x.bw_e is None:
                out_block_max = torch.einsum("bnmij, mpij->bnmpij",x.max_data , mat.max_data)
                out = (out* out_block_max
                        / (2 ** (sum(x.slice_method) - 1) - 1) / (2 ** (sum(mat.slice_method) - 1) - 1))
            else:
                out_block_e_bias = torch.einsum("bnmij, mpij->bnmpij",2.**x.e_bias , 2.**mat.e_bias)
                out = out* out_block_e_bias*2.**(4-sum(x.slice_method)-sum(mat.slice_method))
            out = out.sum(dim=2)
            out = out.permute(0, 1, 3, 2, 4)
            out = out.reshape(out.shape[0],out.shape[1] * out.shape[2], out.shape[3] * out.shape[4])
            result = out[:out.shape[0],:x.shape[1],:mat.shape[1]]
        else:
            raise ValueError('The input data dimension is not supported!')

        return result

    def slice_data(self, mat: torch.Tensor, slice_method: (torch.Tensor, list), bw_e: int = None, slice_data_flag: bool = False):
        """
        Slices the input or weight data using the specified method.

        Parameters:
            mat (torch.Tensor): Data to be sliced.
            slice_method (torch.Tensor): Method for slicing.
            bw_e (int): Bit width for exponent (if None, uses integer format).
            slice_data_flag (bool): Flag for slicing data (True for input, False for weight).

        Returns:
            data_int (torch.Tensor): the quantized data, the shape is (num_divide_row_a, num_divide, num_slice ,m , n) or (batch, num_divide_row_a, num_divide, num_slice ,m , n)
            mat_data (torch.Tensor): the data quantized by the slice method, the shape is the same as the data
            max_mat (torch.Tensor): the max value of the data for each quantization granularity, the shape is (num_divide_row_a, num_divide, 1, 1) or (batch, num_divide_row_a, num_divide, 1, 1)
            e_bias (torch.Tensor): the bias of the exponent for each quantization granularity, the shape is (num_divide_row_a, num_divide, 1, 1) or (batch, num_divide_row_a, num_divide, 1, 1)
        """
        # Convert 2d to 3d makes it easier to follow along with the process
        unsqueezed = False
        if len(mat.shape) == 2:
            mat = mat.unsqueeze(0)
            unsqueezed = True

        quant_gran = self.input_quant_gran if slice_data_flag else self.weight_quant_gran
        paral_size = self.input_paral_size if slice_data_flag else self.weight_paral_size
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
        #extend quant_gran to an integer multiple of paral_size
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
        # Broadcast max_abs_temp_mat from (mat.shape[0], num_gran_row, num_gran_col, 1, 1) to (mat.shape[0], num_gran_row, num_gran_col, num_divide_row, num_divide_col, 1, 1)
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

def dot_high_dim(x, y):
    """
    Computes the dot product of two sliced high-dimensional tensors using torch.einsum.

    Parameters:
        x (torch.Tensor): First tensor (shape: (num_divide_row_x, num_divide_col_x, num_slice_x, m, n) or (batch, num_divide_row_x, num_divide_col_x, num_slice_x, m, n)).
        y (torch.Tensor): Second tensor (shape: (num_divide_row_y, num_divide_col_y, num_slice_y, n, p)).

    Returns:
        torch.Tensor: Result of the dot product (shape: (num_divide_row_x, num_divide_col_y, num_slice_x, num_slice_y, m, p) or (batch, num_divide_row_x, num_divide_col_y, num_slice_x, num_slice_y, m, p)).
    """
    if len(x.shape) == 5:       #if the input data has no batch
        return torch.einsum("nmijk, mpskl->nmpisjl", x, y)
    elif len(x.shape) == 6:     #if the input data has batch
        return torch.einsum("bnmijk, mpskl->bnmpisjl", x, y)
    else:
        raise ValueError('The input data dimension is not supported!')

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
        x = SlicedData(xblk, device=device,bw_e=8,slice_data_flag=True)
        size = 64
        paral_size = size
        engine = DPETensor(var=0.00,g_level=16,rdac=16,radc=2**16,weight_quant_gran=(size,size),input_quant_gran=(1,size),weight_paral_size=(paral_size,paral_size),input_paral_size=(1,paral_size))
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
    elif tb_mode == 1:
        torch.manual_seed(42)
        x_data = torch.randn(1000, 1000,dtype=torch.float64,device=device)
        mat_data = torch.randn(1000,1000,dtype=torch.float64,device=device)
        mblk = torch.tensor([1,1,1])
        xblk = torch.tensor([1,1,1,])
        mat = SlicedData(mblk, device=device,bw_e=None)
        x = SlicedData(xblk, device=device,bw_e=None,slice_data_flag=True)
        size = 256
        paral_size = size
        for radc in [2**4,2**5,2**6,2**7,2**8,2**9,2**10,2**11,2**12]:
            engine = DPETensor(var=0.02,g_level=2,rdac=2,radc=radc,weight_quant_gran=(size,size),input_quant_gran=(1,size),weight_paral_size=(paral_size,paral_size),input_paral_size=(1,paral_size))
            mat.slice_data_imp(engine, mat_data)
            x.slice_data_imp(engine, x_data)
            start = time.time()
            result = engine(x, mat).cpu().numpy()
            rel_result = torch.matmul(x_data, mat_data).cpu().numpy()
            snr_varlue = SNR(result, rel_result)
            print("radc=",radc,"SNR(dB)",snr_varlue)

            plt.scatter(rel_result.reshape(-1), result.reshape(-1))
            plt.xlabel('Expected Value of Dot Product')
            plt.ylabel('Measured Value of Dot Product')
        #plt.show()
        



