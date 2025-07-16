# -*- coding: utf-8 -*-
# =============================================================================
# @File  : 01_matrix_multiplication.py
# @Author: ZZW
# @Date  : 2025/02/11
# @Desc  : Memintelli example 1: Analog CIM-based Matrix Multiplication.
# This example demonstrates how to perform matrix multiplication using a CIM simulation framework. It includes initialization of CIM engines, 
# bit-slicing, and computation of the Signal-to-Noise Ratio (SNR) to evaluate the result.
# =============================================================================
import torch
import numpy as np
from matplotlib import pyplot as plt

from memintelli.pimpy.memmat_tensor import DPETensor
from memintelli.pimpy.data_formats import SlicedData

# Define the Signal-to-Noise Ratio (SNR)
def SNR(p_actual, p_ideal):
    return 10 * np.log10(np.sum(p_actual**2) / np.sum((p_ideal - p_actual)**2))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    # The following codes are to initialize the memristive engine, where the parameters are the same as the memristor crossbar array. 
    mem_engine = DPETensor(
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

    # Initialize input and matrix data
    input_data = torch.randn(400, 500, device=device)
    weight_data = torch.randn(500, 600, device=device)
    # Define dynamic bit-slicing parameters for input and weight
    input_slice = torch.tensor([1, 1, 2, 2, 2])
    weight_slice = torch.tensor([1, 1, 2, 2, 2])

    weight_quant_gran = (128, 128)   # Quantization granularity of the weight matrix
    input_quant_gran = (1, 128)      # Quantization granularity of the input matrix
    weight_paral_size = (32, 32)     # The size of the crossbar array used for parallel computation, 
                                    # where (32, 32) here indicates that the weight matrix is divided into 32x32 sub-arrays for parallel computation
    input_paral_size = (1, 32)        # The size of the input data used for parallel computation,
                                        # where (1, 32) here indicates that the input matrix is divided into 1×32 sub-inputs for parallel computation

    # Create sliced data objects and slice the input and weight data according to the memristive engine's parameters 
    # INT mode
    input_int = SlicedData(input_slice, device=device, bw_e=None, is_weight=False, paral_size=input_paral_size, quant_gran=input_quant_gran)
    weight_int = SlicedData(weight_slice, device=device, bw_e=None, is_weight=True, paral_size=weight_paral_size, quant_gran=weight_quant_gran)
    input_int.slice_data_imp(mem_engine,input_data)
    weight_int.slice_data_imp(mem_engine,weight_data)
    # FP mode
    input_fp = SlicedData(input_slice, device=device, bw_e=8, is_weight=False, paral_size=input_paral_size, quant_gran=input_quant_gran)
    weight_fp = SlicedData(weight_slice, device=device, bw_e=8, is_weight=True, paral_size=weight_paral_size, quant_gran=weight_quant_gran)
    input_fp.slice_data_imp(mem_engine,input_data)
    weight_fp.slice_data_imp(mem_engine,weight_data)

    # Perform matrix multiplication using software and the memristive engine with INT and FP modes. The functions are equivalent to torch.matmul(input_data, weight_data) 
    result_ideal = torch.matmul(input_data, weight_data).cpu().numpy()
    result_int = mem_engine(input_int, weight_int).cpu().numpy()
    result_fp = mem_engine(input_fp, weight_fp).cpu().numpy()
    
    # Calculate the Signal-to-Noise Ratio (SNR) of the result and plot the scatter plot of the expected and measured values 
    snr_int = SNR(result_int, result_ideal)
    snr_fp = SNR(result_fp, result_ideal)
    print(f"Signal Noise Ratio (SNR): {snr_int:.2f} dB")
    print(f"Signal Noise Ratio (SNR): {snr_fp:.2f} dB")

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(result_int.reshape(-1), result_ideal.reshape(-1))
    plt.title(f"INT Mode (SNR: {snr_int:.2f} dB)")
    plt.xlabel('Ideal Result of matrix multiplication')
    plt.ylabel('Actual Result of matrix multiplication')

    plt.subplot(1, 2, 2)
    plt.scatter(result_fp.reshape(-1), result_ideal.reshape(-1))
    plt.title(f"FP Mode (SNR: {snr_fp:.2f} dB)")
    plt.xlabel('Ideal Result of matrix multiplication')
    plt.ylabel('Actual Result of matrix multiplication')

    plt.show()
    
if __name__ == "__main__":
    main()