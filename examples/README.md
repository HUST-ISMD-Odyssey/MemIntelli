# Memintelli---Memristive Intelligient Computing Simulator: Examples
We have many different examples to explore the many features of Memristive Intelligient Computing Simulator.
## Example 1: [`01_matrix_multiplication.py`](./01_matrix_multiplication.py)
In this example, a simple matrix multiplication is demonstrated by Memintelli. It specifies the memristive device parameter of `DPETensor` as follows:

```python
mem_engine = DPETensor(
        HGS=1e-5,                       # High conductance state
        LGS=1e-8,                       # Low conductance state
        var=0.05,                       # Random Gaussian noise of conductance
        rdac=2**2,                      # Number of DAC resolution 
        g_level=2**2,                   # Number of conductance levels
        radc=2**10,                     # Number of ADC resolution 
        weight_quant_gran=(128, 128),   # Quantization granularity of the weight matrix
        input_quant_gran=(1, 128),      # Quantization granularity of the input matrix
        weight_paral_size=(32, 32),     # The size of the crossbar array used for parallel computation, 
                                        # where (32, 32) here indicates that the weight matrix is divided into 32x32 sub-arrays for parallel computation
        input_paral_size=(1, 32)        # The size of the input data used for parallel computation,
                                        # where (1, 32) here indicates that the input matrix is divided into 1×32 sub-inputs for parallel computation
    )
```
And the dynamic bit-slicing method is used for inputs and weights, which can combine the accuracy of SLC (single-level cell) and the efficiency of MlC (multi-level cell), and the details can be viewed on the paper: [Shao-Qin Tong et al. (2024) Energy-Efficient Brain Floating Point Convolutional Neural Network Using Memristors. IEEE TED](https://ieeexplore.ieee.org/abstract/document/10486875).
```python
# Define dynamic bit-slicing parameters for input and weight. Inputs and weights both use 8-bits, where the higher two bits use two SLCs and the remaining bits consist of 3 MLCs
input_slice = torch.tensor([1, 1, 2, 2, 2]) 
weight_slice = torch.tensor([1, 1, 2, 2, 2])
```
Here we use both INT quantization and FP quantization modes, of which the principles are described in the paper: [Zhiwei Zhou et al. (2024) ArPCIM: An Arbitrary-Precision Analog Computing-in-Memory Accelerator With Unified INT/FP Arithmetic. IEEE TCAS-I](https://ieeexplore.ieee.org/abstract/document/10486875).

In INT mode, `bw_e` should be set to `None` and `slice_data_flag` should be set to `True` for input `SlicedData`.
```python
input_int = SlicedData(input_slice, device=device, bw_e=None, slice_data_flag=True)
weight_int = SlicedData(weight_slice, device=device, bw_e=None)
input_int.slice_data_imp(mem_engine,input_data)
weight_int.slice_data_imp(mem_engine,weight_data)
```
While in FP mode, `bw_e` should be set to bit width of the exponent bit (e.g. 8 bits for BF16, 5 bits for FP16) and `slice_data_flag` should be set to `True` for input `SlicedData`.
```python
input_fp = SlicedData(input_slice, device=device, bw_e=8, slice_data_flag=True)
weight_fp = SlicedData(weight_slice, device=device, bw_e=8)
input_fp.slice_data_imp(mem_engine,input_data)
weight_fp.slice_data_imp(mem_engine,weight_data)
```
We use SNR (signal to nosie ratio) to describe the error between ideal result and the actual result, and plot a distribution graph for both cases.

![Test Losses](./img/SNR_of_INT_and_FP.png)

## Example 2: [`02_MLP_inference.py`](./02_MLP_inference.py)
The second example uses a single mlp classifier with MNIST dataset. The defined `MNISTClassifier` adds the parameters `engine`, `input_slice`, `weight_slice`, `bw_e`, `mem_enabled` to the traditional nn.Module. Among them, `engine` is the same as in [`Example 1`](./01_matrix_multiplication.py), `input_slice` and `weight_slice` correspond to the method of dynamic bit-slicing, and `bw_e` determines whether to use INT or FP mode. 