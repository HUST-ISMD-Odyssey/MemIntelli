"""Optional PyTorch NVRTC kernels; no Triton or external CUDA compiler."""
import math
from functools import lru_cache
import warnings

import torch


@lru_cache(maxsize=32)
def _compile(source, name, device):
    with torch.cuda.device(device):
        return torch.cuda._compile_kernel(source, name, nvcc_options=["--fmad=false"])


def get_kernel(engine, name):
    if getattr(engine, "_torch_cuda_fusion_error", None) is not None:
        return None
    attribute = "_torch_cuda_" + name
    if hasattr(engine, attribute):
        return getattr(engine, attribute)
    builder = {"adc_reduce": compile_adc_reduce, "voltage": compile_voltage,
               "slice": compile_slice}[name]
    try:
        kernel = builder(engine)
    except (AttributeError, ImportError, OSError, RuntimeError, TypeError) as exc:
        if isinstance(exc, torch.cuda.OutOfMemoryError) or any(
            text in str(exc).lower() for text in ("illegal memory access", "device-side assert")
        ):
            raise
        engine._torch_cuda_fusion_error = "\n".join(str(exc).splitlines()[-3:])[-1000:]
        engine._torch_cuda_fusion_status = "unavailable"
        warnings.warn("Torch grouped CUDA fusion unavailable; retaining the portable path: "
                      + engine._torch_cuda_fusion_error, RuntimeWarning, stacklevel=3)
        return None
    setattr(engine, attribute, kernel)
    return kernel


def compile_adc_reduce(engine):
    rows = engine.weight_paral_size[0]
    ni, ns = len(engine.input_slice), len(engine.weight_slice)
    exponent = max(1, math.ceil(math.log2(rows)))
    step = 2**(exponent-engine.adc_bits)
    highest = min(2**engine.adc_bits-1, math.floor(min(rows, 2**exponent-1)/step))
    inverse = torch.tensor((engine.HGS-engine.LGS)*engine.vread*rows,
                           dtype=torch.float32).reciprocal().item()
    xinv = torch.tensor(2**(ni-1)-1, dtype=torch.float32).reciprocal().item()
    winv = torch.tensor(2**(ns-1)-1, dtype=torch.float32).reciprocal().item()
    source = f'''
extern "C" __global__ void memintelli_adc_reduce(
    const unsigned short* currents, const float* xm, const float* wm, float* output,
    int blocks, int chunk, int samples, int columns, int begin) {{
    long long index = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= (long long)blocks*chunk*columns) return;
    int col = index % columns;
    int row = (index / columns) % chunk;
    int block = index / ((long long)chunk*columns);
    float result = 0.0f;
    #pragma unroll
    for (int i = 0; i < {ni}; ++i) {{
        #pragma unroll
        for (int s = 0; s < {ns}; ++s) {{
            long long address = ((((long long)block*{ni}+i)*chunk+row)*{ns}+s)*columns+col;
            float current = __uint_as_float((unsigned int)currents[address] << 16);
            float q = nearbyintf(__fmul_rn(__fmul_rn(__fmul_rn(current, {inverse}f),
                                          {float(rows)}f), {1.0/step}f));
            q = q < 0.0f ? 0.0f : (q > {float(highest)}f ? {float(highest)}f : q);
            float sign = float((i == {ni-1} ? -(1 << i) : (1 << i)) *
                               (s == {ns-1} ? -(1 << s) : (1 << s)));
            result = __fadd_rn(result, __fmul_rn(__fmul_rn(q, {float(step)}f), sign));
        }}
    }}
    float scale = __fmul_rn(xm[(long long)(begin+row)*blocks+block],
                           wm[(long long)block*columns+col]);
    result = __fmul_rn(__fmul_rn(__fmul_rn(result, scale), {xinv}f), {winv}f);
    output[((long long)block*samples+begin+row)*columns+col] = result;
}}
'''
    return _compile(source, "memintelli_adc_reduce", str(engine.device))


def compile_voltage(engine):
    bits = int(torch.tensor(engine.vread, dtype=torch.float32).to(torch.bfloat16).view(torch.int16)) & 65535
    ni, k = len(engine.input_slice), engine.weight_paral_size[0]
    source = f'''
extern "C" __global__ void memintelli_voltage(
    const unsigned char* x, unsigned short* v, int blocks, int chunk, int begin,
    int stride0, int stride1, int stride2, int stride4) {{
    long long index = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (index >= (long long)blocks*{ni}*chunk*{k}) return;
    int col = index % {k};
    int row = (index / {k}) % chunk;
    int slice = (index / ((long long){k}*chunk)) % {ni};
    int block = index / ((long long){k}*chunk*{ni});
    long long address = (long long)(begin+row)*stride0 + (long long)block*stride1
                        + (long long)slice*stride2 + (long long)col*stride4;
    v[index] = x[address] ? {bits} : 0;
}}
'''
    return _compile(source, "memintelli_voltage", str(engine.device))


def compile_slice(engine):
    widths = tuple(reversed(engine.input_slice))
    shifts, offset = [], 0
    for width in widths:
        shifts.append(str(offset))
        offset += width
    masks = ",".join(str(2**width-1) for width in widths)
    ni, size, group = len(widths), engine.weight_paral_size[0], engine.input_quant_gran[1]
    source = f'''
extern "C" __global__ void memintelli_slice(
    const float* x, const float* maxima, unsigned char* bits,
    int rows, int features, int blocks, int groups, int stride0, int stride1) {{
    long long index = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (index >= (long long)rows*blocks*{size}) return;
    const int shifts[{ni}] = {{{",".join(shifts)}}};
    const int masks[{ni}] = {{{masks}}};
    int col = index % ((long long)blocks*{size});
    int row = index / ((long long){size}*blocks);
    float value = col < features ? x[(long long)row*stride0+(long long)col*stride1] : 0.0f;
    float maximum = maxima[(long long)row*groups+col/{group}];
    float scaled = __fmul_rn(__fdiv_rn(value, maximum > 0.0f ? maximum : 1.0f),
                            {float(2**(engine.activation_bits-1)-1)}f);
    int quantized = __float2int_rn(scaled);
    #pragma unroll
    for (int slice = 0; slice < {ni}; ++slice) {{
        long long address = (((long long)row*blocks+col/{size})*{ni}+slice)*{size}+col%{size};
        bits[address] = (quantized >> shifts[slice]) & masks[slice];
    }}
}}
'''
    return _compile(source, "memintelli_slice", str(engine.device))
