# -*- coding:utf-8 -*-
# @File  : dpe_tensor.py
# @Author: Zhou
# @Date  : 2024/6/27

'''
this is a new version of the dpe_tensor.py
we use the tensor to realize the dot product, and only consider the INT format data
this version is more efficient than the previous version
'''
import math

import torch
from matplotlib import pyplot as plt
from memintelli.pimpy.utils import SNR, dot_high_dim
from memintelli.pimpy.data_formats import SlicedData

import time


def wire_resistance(x, y, wire_resistance):
    pass


class DPETensor(object):
    '''
    Implements a dot product engine using bit-sliced tensor operations for matrix multiplication.
    Supports INT and FP data formats with configurable quantization granularity and device settings.
    '''
    def __init__(
            self, HGS=1e-5, LGS=1e-7, g_level=16, write_variation=0.02, read_variation=0.02,
             vnoise=0.05, wire_resistance=0,
            rdac=2 ** 4, radc=2 ** 8, vread=0.2,
            rate_stuck_HGS=0.001, rate_stuck_LGS=0.000,
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        """
        Parameters:
            HGS (float): High conductance state
            LGS (float): Low conductance state
            g_level (int): Number of conductance levels
            write_variation (float): Write variation - fixed Gaussian noise added to each conductance state
            read_variation (dict or float): Read variation - dynamic noise for each g level
                Can be a dict like {0: 0.05, 1: 0.1, 2: 0.1, 3: 0.15} or a single float applied to all levels
            vnoise (float): Random Gaussian noise of voltage
            wire_resistance (float): Wire resistance
            rdac (int): Number of DAC resolution
            radc (int or list): Number of ADC resolution. Can be:
                - Single integer: Same resolution for all weight slices
                - List/tuple: Different resolution for each weight slice (length must match number of weight slices)
            vread (float): Read voltage
            rate_stuck_HGS (float): Ratio of stuck faults to HGS state, i.e., the probability of stuck on, range: [0, 1]
            rate_stuck_LGS (float): Ratio of stuck faults to LGS state, i.e., the probability of stuck off, range: [0, 1]
            weight_quant_gran (str or tuple): Quantization granularity of the weight matrix
                "per-matrix" -> The whole matrix is quantized together (i.e., the quantization granularity is (m, n)
                                the same as the matrix shape).
                "per-row" -> Each row of the matrix is quantized separately. (i.e., the quantization granularity is (1, n)).
                "per-col" -> Each column of the matrix is quantized separately. (i.e., the quantization granularity is (m, 1)).
                (a, b) -> The quantization granularity is (a, b).
            input_quant_gran (str or tuple): Quantization granularity of the input matrix
            input_paral_size (tuple): The size of the input matrix used for parallel computation
            weight_paral_size (tuple): The size of the weight matrix used for parallel computation
        """
        self.HGS = HGS
        self.LGS = LGS
        self.g_level = g_level
        self.write_variation = write_variation
        
        # Handle read variation parameter
        if isinstance(read_variation, dict):
            # Validate the dictionary keys
            if set(read_variation.keys()) != set(range(g_level)):
                raise ValueError(f'read_variation dict must have keys for all g levels 0 to {g_level-1}')
            self.read_variation = read_variation
        elif isinstance(read_variation, (int, float)):
            # Single value applied to all levels
            self.read_variation = {i: read_variation for i in range(g_level)}
        else:
            raise ValueError('read_variation must be a dict, float, or None')

        self.vnoise = vnoise
        self.rdac = rdac
        if isinstance(radc, (list, tuple)):
            self.radc = torch.tensor(radc, device=device)
            self.radc = self.radc.flip(0)
            self.radc_is_list = True
        else:
            self.radc = radc
            self.radc_is_list = False
        self.vread = vread

        # these parameters are optional
        self.wire_resistance = wire_resistance
        self.rate_stuck_HGS = rate_stuck_HGS
        self.rate_stuck_LGS = rate_stuck_LGS

        self.device = device

        if self.radc_is_list:
            if torch.any(self.radc < 2):
                raise ValueError('All ADC resolution values should be larger than 1!')
        else:
            if self.radc < 2:
                raise ValueError('The resolution of the ADC should be larger than 1!')
        if self.rdac < 2:
            raise ValueError('The resolution of the DAC should be larger than 1!')
        if self.g_level < 2:
            raise ValueError('The number of the conductance levels should be larger than 1!')
        if self.LGS >= self.HGS:
            raise ValueError('The low conductance state should be smaller than the high conductance state!')
        if self.rate_stuck_HGS + self.rate_stuck_LGS > 1:
            raise ValueError('The sum of stuck fault rates should not exceed 1!')

        # Pre-compute conductance levels for efficiency
        self.Q_G = (self.HGS - self.LGS) / (self.g_level - 1)
        self.conductance_levels = torch.tensor([self.LGS + i * self.Q_G for i in range(self.g_level)], device=self.device)

    def _validate_radc_with_slices(self, mat: SlicedData):
        """
        Validate that radc array length matches the number of weight slices.
        
        Parameters:
            mat (SlicedData): Weight tensor data.
        """
        if self.radc_is_list:
            num_weight_slices = len(mat)
            if len(self.radc) != num_weight_slices:
                raise ValueError(f'Length of radc array ({len(self.radc)}) must match number of weight slices ({num_weight_slices})')

    def __call__(self, x: SlicedData, mat: SlicedData):
        return self.MapReduceDot(x, mat)

    def MapReduceDot(self, x: SlicedData, mat: SlicedData):
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
        # check the quantization shape of the input data and weight data
        if x.shape[-1] != mat.shape[-2]:
            raise ValueError('The input data mismatches the shape of weight data!')
        self._validate_radc_with_slices(mat)
        if self.wire_resistance > 0:
            raise NotImplementedError('The wire_factor is not supported in the training version!')
        else:
            result = self._dot(x, mat, self._num2V, self._gen_read_noise)
        return result

    def _num2G(self, data, max_weights):
        """
        Converts weight data to static resistance.
        Parameters:
            data (torch.Tensor): Weight data.
            max_weights (torch.Tensor): Maximum weight values.

        Returns:
            torch.Tensor: conductance values.
        """
        # Step 1: Quantization to conductance levels
        level_indices = torch.round(data / max_weights * (self.g_level - 1)).long()
        G = level_indices * self.Q_G + self.LGS

        # Step 2: Add write variation (fixed noise per device)
        if self.write_variation > 0:
            # fixed seed for write variation
            generator = torch.Generator(device=G.device)
            generator.manual_seed(42)  # fixed seed 
            write_noise = torch.normal(0, self.write_variation, G.shape, generator=generator, device=G.device)
            G = G + G * write_noise
        
        # Step 3: Add stuck at fault (applied after write variation)
        if self.rate_stuck_HGS > 0 or self.rate_stuck_LGS > 0:
            # fixed seed for stuck fault
            generator_stuck = torch.Generator(device=G.device)
            generator_stuck.manual_seed(123)  # fixed seed
            random_vals = torch.rand(G.shape, generator=generator_stuck, device=G.device)
            
            # Apply stuck at HGS
            if self.rate_stuck_HGS > 0:
                stuck_hgs_mask = random_vals < self.rate_stuck_HGS
                G[stuck_hgs_mask] = self.HGS
            
            # Apply stuck at LGS
            if self.rate_stuck_LGS > 0:
                stuck_lgs_mask = (random_vals >= self.rate_stuck_HGS) & (random_vals < self.rate_stuck_HGS + self.rate_stuck_LGS)
                G[stuck_lgs_mask] = self.LGS
            
        # add the wire resistance
        if self.wire_resistance > 0:
            pass
        return G

    def _gen_read_noise(self, mat: SlicedData):
        """
        Converts weight data to resistance with added normal noise.

        Parameters:
            mat (SlicedData): Weight data.

        Returns:
            torch.Tensor: Resistance values.
        """
        G = mat.G
        
        # Apply read variation based on g level
        if any(var > 0 for var in self.read_variation.values()):
            # Determine which g level each conductance value belongs to
            # Find the closest conductance level for each value
            expanded_G = G.unsqueeze(-1)  # Add dimension for broadcasting
            
            # Reshape conductance levels to match the expanded G dimensions
            num_dims = len(G.shape)
            reshape_dims = [1] * num_dims + [-1]
            expanded_levels = self.conductance_levels.view(reshape_dims)
            
            # Find closest level index
            distances = torch.abs(expanded_G - expanded_levels)
            closest_level_idx = torch.argmin(distances, dim=-1)
            
            # Apply read variation for each g level
            noise = torch.zeros_like(G)
            for level_idx, variation in self.read_variation.items():
                if variation > 0:
                    level_mask = (closest_level_idx == level_idx)
                    level_noise = torch.normal(0, variation, G.shape, device=G.device)
                    noise[level_mask] = level_noise[level_mask]
            
            # Apply multiplicative noise (similar to original implementation)
            r = torch.exp(noise)
            G = G * r
        
        return G

    def _num2V(self, x: SlicedData):
        """
        Converts input data to voltage (scaled by read voltage).

        Parameters:
            x (SlicedData): Input data.

        Returns:
            torch.Tensor: Voltage values.
        """
        xmax = x.sliced_max_weights
        if len(x.shape) == 2:  # without batch, the shape is (num_divide_row_x, num_divide_col_x, num_slice_x, m, n)
            xmax = xmax.reshape(1, 1, -1, 1, 1)
        elif len(x.shape) == 3:  # with batch, the shape is (batch, num_divide_row_x, num_divide_col_x, num_slice_x, m, n)
            xmax = xmax.reshape(1, 1, 1, -1, 1, 1)
        else:
            raise ValueError('The input data dimension is not supported!')
        V_in = self.vread * torch.round(x.sliced_data / xmax * (self.rdac - 1)) / (self.rdac - 1)
        V_in = V_in * (1+ torch.normal(0, self.vnoise, V_in.shape, device=V_in.device))
        return V_in

    def _dot(self, x: SlicedData, mat: SlicedData, _num2V_func, _num2R_func):
        """
        Computes the dot product of input and weight tensors.

        Parameters:
            x (SlicedData): Input tensor with shape (m, n) or (batch, m, n).
            mat (SlicedData): Weight tensor with shape (n, p).
            _num2V_func (function): Function to convert input data to voltage.
            _num2R_func (function): Function to convert weight data to resistance

        Returns:
            torch.Tensor: Result of the dot product with shape (m, p) or (batch, m, p).
        """
        Vin = _num2V_func(x)
        G = _num2R_func(mat)
        
        if max(mat.sliced_max_weights) > self.g_level - 1:
            raise ValueError('The weight data is out of the range!')

        if len(x.shape) == 2:
            adcRef = (self.HGS - self.LGS) * self.vread * Vin.shape[-1]
            QG = (self.HGS - self.LGS) / (self.g_level - 1)
            out = dot_high_dim(Vin, G - self.LGS)
            if self.radc_is_list:
                radc_expanded = self.radc.view(1, 1, 1, 1, -1, 1, 1)  # reshape to broadcast correctly
                out = torch.round(out / adcRef * (radc_expanded - 1)) / (radc_expanded - 1)
            else:
                out = torch.round(out / adcRef * (self.radc - 1)) / (self.radc - 1)

            out = torch.mul(out, x.sliced_max_weights.reshape(1, 1, 1, -1, 1, 1, 1))
            out = (torch.mul(out, mat.sliced_max_weights.reshape(1, 1, 1, 1, -1, 1, 1)) / QG / self.vread / (
                        self.g_level - 1) * adcRef)
            shift_weights = torch.zeros((len(x), len(mat)), device=x.device)

            for i in range(len(x)):
                shift_weights[i] = x.sliced_weights[i] * mat.sliced_weights
            out = torch.mul(out.reshape(out.shape[0], out.shape[1], out.shape[2], -1, out.shape[5], out.shape[6]),
                            shift_weights.reshape(1, 1, 1, -1, 1, 1))
            out = out.sum(dim=3)
            if x.bw_e is None:
                out_block_max = torch.einsum("nmij, mpij->nmpij", x.max_data, mat.max_data)
                out = (out * out_block_max / (2 ** (sum(x.slice_method) - 1) - 1) / (
                            2 ** (sum(mat.slice_method) - 1) - 1))
            else:
                out_block_e_bias = torch.einsum("nmij, mpij->nmpij", 2. ** x.e_bias, 2. ** mat.e_bias)
                out = out * out_block_e_bias * 2. ** (4 - sum(x.slice_method) - sum(mat.slice_method))
            out = out.sum(dim=1)
            out = out.permute(0, 2, 1, 3)
            out = out.reshape(out.shape[0] * out.shape[1], out.shape[2] * out.shape[3])
            out = out[:x.shape[0], :mat.shape[1]]
        elif len(x.shape) == 3:  
            adcRef = (self.HGS - self.LGS) * self.vread * Vin.shape[-1]
            QG = (self.HGS - self.LGS) / (self.g_level - 1)
            out = dot_high_dim(Vin, G - self.LGS)

            if self.radc_is_list:
                radc_expanded = self.radc.view(1, 1, 1, 1, 1, -1, 1, 1)  # reshape to broadcast correctly
                out = torch.round(out / adcRef * (radc_expanded - 1)) / (radc_expanded - 1)
            else:
                out = torch.round(out / adcRef * (self.radc - 1)) / (self.radc - 1)
            out = torch.mul(out, x.sliced_max_weights.reshape(1, 1, 1, 1, -1, 1, 1, 1))
            out = (torch.mul(out, mat.sliced_max_weights.reshape(1, 1, 1, 1, 1, -1, 1, 1)) / QG / self.vread / (
                        self.g_level - 1) * adcRef)
            shift_weights = torch.zeros((len(x), len(mat)), device=x.device)

            for i in range(len(x)):
                shift_weights[i] = x.sliced_weights[i] * mat.sliced_weights
            # add the shift weights to the calculated result
            out = torch.mul(out.reshape(out.shape[0], out.shape[1], out.shape[2], out.shape[3], -1, out.shape[6],
                                         out.shape[7]), shift_weights.reshape(1, 1, 1, 1, -1, 1, 1))
            out = out.sum(dim=4)
            if x.bw_e is None:
                out_block_max = torch.einsum("bnmij, mpij->bnmpij", x.max_data, mat.max_data)
                out = (out * out_block_max / (2 ** (sum(x.slice_method) - 1) - 1) / (
                            2 ** (sum(mat.slice_method) - 1) - 1))
            else:
                out_block_e_bias = torch.einsum("bnmij, mpij->bnmpij", 2. ** x.e_bias, 2. ** mat.e_bias)
                out = out * out_block_e_bias * 2. ** (4 - sum(x.slice_method) - sum(mat.slice_method))
            out = out.sum(dim=2)
            out = out.permute(0, 1, 3, 2, 4)
            out = out.reshape(out.shape[0], out.shape[1] * out.shape[2], out.shape[3] * out.shape[4])
            out = out[:out.shape[0], :x.shape[1], :mat.shape[1]]
        else:
            raise ValueError('The input data dimension is not supported!')
        return out


if __name__ == '__main__':
    tb_mode = 1
    device = torch.device('cuda:0')
    if tb_mode == 0:
        torch.manual_seed(42)
        x_data = torch.randn(2, 1000, 1000, dtype=torch.float64, device=device)
        mat_data = torch.randn(1000, 1000, dtype=torch.float64, device=device)
        mblk = torch.tensor([1, 1, 2, 4])
        xblk = torch.tensor([1, 1, 2, 4])
        mat = SlicedData(mblk, device=device, bw_e=None, is_weight=True, quant_gran=(64, 64), paral_size=(64, 64))
        x = SlicedData(xblk, device=device, bw_e=None, quant_gran=(64, 64), paral_size=(64, 64))
        engine = DPETensor(write_variation=0.0, read_variation=0.0, rate_stuck_HGS=0.0, rate_stuck_LGS=0.0, vnoise=0.01,  g_level=16, rdac=16, radc=2 ** 16)
        mat.slice_data_imp(engine, mat_data)
        x.slice_data_imp(engine, x_data)
        start = time.time()
        result = engine(x, mat).cpu().numpy()
        rel_result = torch.matmul(x_data, mat_data).cpu().numpy()
        snr_varlue = SNR(result, rel_result)
        print("SNR(dB)", snr_varlue)
        plt.scatter(rel_result.reshape(-1), result.reshape(-1))
        plt.xlabel('Expected Value of Dot Product')
        plt.ylabel('Measured Value of Dot Product')
        #plt.show()
    elif tb_mode == 1:
        torch.manual_seed(42)
        x_data = torch.randn(1000, 1000, dtype=torch.float64, device=device)
        mat_data = torch.randn(1000, 1000, dtype=torch.float64, device=device)
        mblk = torch.tensor([1, 1, 2, 2])
        xblk = torch.tensor([1, 1, 2, 2])
        for i in range(10):
            mat = SlicedData(mblk, device=device, bw_e=None, is_weight=True, quant_gran=(64, 64), paral_size=(64, 64))
            x = SlicedData(xblk, device=device, bw_e=None, quant_gran=(64, 64), paral_size=(64, 64))
            read_variation = {0: 0.05, 1: 0.05, 2: 0.05, 3: 0.1}
            radc_per_slice = [2**12, 2**11, 2**10, 2**9]  # Different radc for each slice
            engine = DPETensor(write_variation=0.0, read_variation=read_variation, rate_stuck_HGS=0.0, rate_stuck_LGS=0.0, vnoise=0.0,  
            g_level=4, rdac=4, radc=radc_per_slice)
            mat.slice_data_imp(engine, mat_data)
            x.slice_data_imp(engine, x_data)
            start = time.time()
            result = engine(x, mat).cpu().numpy()
            rel_result = torch.matmul(x_data, mat_data).cpu().numpy()
            snr_varlue = SNR(result, rel_result)
            print("SNR(dB)", snr_varlue)
            plt.scatter(rel_result.reshape(-1), result.reshape(-1))
            plt.xlabel('Expected Value of Dot Product')
            plt.ylabel('Measured Value of Dot Product')