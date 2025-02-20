# Memintelli---Memristive Intelligient Computing Simulator: Examples
We have many different examples to explore the many features of Memristive Intelligient Computing Simulator.
## Example 1: [01_matrix_multiplication.py](./01_matrix_multiplication.py)
In this example, a simple matrix multiplication is demonstrated by Memintelli. It specifies the memristive device parameter of `DPETensor` as follows:

```
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