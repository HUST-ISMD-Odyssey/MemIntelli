# Memintelli: A Quantitative Error Modeling Framework for General-Purpose In-Memory Computing

## Description

_Memintelli_ is an open source Python-based framework, that provides an important priori guidance for hardware design in the field of general-purpose in-memory computing. 

## Installation
1. Get the tool from GitHub
```
git clone https://github.com/HUST-ISMD-Odyssey/Memintelli.git
```
2. Install by 
```
cd Memintelli
pip install .
```
3. (optional) Installing the necessary packages
```
pip install numpy==1.26 matplotlib tqdm
```
* `numpy` needs to be installed version `<2.0.0` due to compatibility issues. 

* For `torch`, `torchvision` and `torchaudio`, it's best to install your own version of your CUDA from https://pytorch.org/get-started/previous-versions/. 

## Usage
### Matrix Multiplication example
```python
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

    weight_quant_gran = (128, 128)   # Quantization granularity of the weight matrix
    input_quant_gran = (1, 128)      # Quantization granularity of the input matrix
    weight_paral_size = (32, 32)     # The size of the crossbar array used for parallel computation, 
                                    # where (32, 32) here indicates that the weight matrix is divided into 32x32 sub-arrays for parallel computation
    input_paral_size = (1, 32)        # The size of the input data used for parallel computation,
                                        # where (1, 32) here indicates that the input matrix is divided into 1×32 sub-inputs for parallel computation

    # Initialize input and matrix data
    input_data = torch.randn(400, 500, device=device)
    weight_data = torch.randn(500, 600, device=device)

    # Define dynamic bit-slicing parameters for input and weight
    input_slice = torch.tensor([1, 1, 2, 2, 2])
    weight_slice = torch.tensor([1, 1, 2, 2, 2])

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
```
You can find more examples in the <u>[`examples`](./examples)</u> folder of the project. 

## Todo list
- [ ] Non-DNN applications based on matrix multiplication (e.g., `signal transformation`, `scientific computing`, `similarity computation`, `combinatorial optimization`)
- [ ] `PTQ` support. 
- [ ] `Mixed-precision` (per-layer, per-channel, per-array, per-block) support.
   

## Contributors

- Houji Zhou, [1499403578@qq.com](mailto:1499403578@qq.com), [zhouhouji (houjizhou) · GitHub](https://github.com/zhouhouji)

- Zhiwei Zhou, [1548384176@qq.com](mailto:1548384176@qq.com)

- Yuyang Fu, [412983100@qq.com](mailto:412983100@qq.com)

## Maintainer

- Maintainer: Researchers from Prof. Xiangshui Miao and [Prof.Yi Li's group](http://ismd.hust.edu.cn/info/1077/1257.htm) at HUST (Huazhong University of Science and Technology). The model is made publicly available on a non-commercial basis.

- Affiliation: Huazhong University of Science and Technology, School of Integrated Circuit,  [Institute of Information Storage Materials and Devices (hust.edu.cn)](http://ismd.hust.edu.cn/)

Any advice and criticism are highly appreciated on this package. Naturally, you can also modify the source code to suit your needs. In upcoming versions, we plan to continually incorporate the latest research findings into Memintelli. 

## References related to this tool
1. Zhiwei Zhou, Jiancong Li, Han Jia, Ling Yang, Houji Zhou, Han Bao, Yuyang Fu, Yi Li*, Xiangshui Miao, ArPCIM: An Arbitrary-Precision Analog Computing-in-Memory Accelerator with unified INT/FP Arithmetic,  IEEE Transactions on Circuits and Systems I: Regular Papers , 2024.  DOI: 10.1109/TCSI.2024.3491825.
2. Yangyu Fu, Jiancong Li, et al. , ReSMiPS: A ReRAM-based Sparse Mixed-precision Solver with Fast Matrix Reordering Algorithm, DAC 2025. DOI: 10.1109/DAC63849.2025.11133301
3. Houji Zhou, Ling Yang, et al. MemIntelli: A Generic End-to-End Simulation Framework for Memristive Intelligent, arXiv preprint arXiv:2511.17418.

