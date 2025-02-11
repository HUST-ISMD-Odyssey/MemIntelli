# -*- coding: utf-8 -*-
# =============================================================================
# @File  : 01_matrix_multiplication.py
# @Author: ZZW
# @Date  : 2025/02/09
# @Desc  : Memintelli example 1: ACIM-based Matrix Multiplication (VMM) .
# =============================================================================
"""Memintelli example 1: Multiple layer inference with MLP on MNIST.
This example demonstrates the usage of Memintelli with a simple MLP classifier that has been trained in software.
"""
import os
import sys
import torch
from matplotlib import pyplot as plt
# Add project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pimpy.memmat_tensor import DPETensor
from utils.data_formats import SlicedData
from utils.functions import SNR

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    x_data = torch.randn(400, 500, device=device)
    mat_data = torch.randn(500, 600, device=device)
    x_slice = torch.tensor([1, 1, 2, 4])
    mat_slice = torch.tensor([1, 1, 2, 4])
    x = SlicedData(x_slice, device=device, bw_e=None)
    mat = SlicedData(mat_slice, device=device, bw_e=None)
      
    # Initialize memory engine and model
    mem_engine = DPETensor(
        var=0.05,
        rdac=2**4,
        g_level=2**4,
        radc=2**12,
        quant_array_gran=(128, 128),
        quant_input_gran=(1, 128),
        paral_array_size=(32, 32),
        paral_input_size=(1, 32)
    )
    x.slice_data_imp(mem_engine,x_data)
    mat.slice_data_imp(mem_engine,mat_data)

    result = mem_engine(x, mat).cpu().numpy()
    ref_result = torch.matmul(x_data, mat_data).cpu().numpy()
    snr_value = SNR(result, ref_result)
    print(f"Signal Noise Ratio (SNR): {snr_value:.2f} dB")

    plt.scatter(ref_result.reshape(-1), result.reshape(-1))
    plt.xlabel('Expected Value of VMM')
    plt.ylabel('Measured Value of VMM')
    plt.show()
    
if __name__ == "__main__":
    main()