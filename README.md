# MemIntelli

[中文入门指南](README.zh-CN.md) | [Runnable examples](examples/README.md)

MemIntelli simulates neural-network computation on bit-sliced memristor arrays.
It models weight quantization, conductance mapping, array currents, ADC
conversion, device variation and digital output reconstruction.

The default is **speed mode with automatic backend selection**: try Triton on
CUDA, then use PyTorch if Triton is unavailable or fails. A fallback emits a
warning and is recorded in the engine status. PyTorch can also be selected
directly, including on native Windows.

## Start Here

MemIntelli is an **array-behavior simulator**, not an INT6 inference accelerator.
A6W6 means six-bit activations and six-bit weights. It does not mean the GPU
executes a native six-bit convolution. The default bit-sliced implementation
evaluates 6 x 6 slice pairs and their ADC conversions, so it is slower than an
ordinary PyTorch model.

Suggested first steps:

1. Install the base package and run the matrix example without any dataset.
2. Check the printed `backend_used`, `device`, `mode` and `calls` fields.
3. Run two samples from one network example with `--limit 2`.
4. Compare `--digital` with array simulation before adding device variation.
5. Remove `--limit` only when the short run, dataset paths and runtime are satisfactory.

The base package supports CPU matrix simulation; network examples need the
corresponding extras listed below. Downloads are stored in the user's cache,
not in this repository.

## Installation

Python 3.10 or newer is required. Python 3.12 is recommended. Create a separate
environment and install a matching PyTorch, torchvision and torchaudio set.

```bash
git clone --branch v2 https://github.com/HUST-ISMD-Odyssey/MemIntelli.git
cd MemIntelli
python -m venv .venv
```

Activate with `.venv\Scripts\activate` on Windows, or
`source .venv/bin/activate` on Linux.

For an NVIDIA GPU, this is the pinned PyTorch 2.9.1 / CUDA 12.6 configuration:

```bash
python -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e ".[vision,speech,llm,yolo]"
```

For CPU-only use, replace the PyTorch index with
`https://download.pytorch.org/whl/cpu`. For Linux Triton acceleration:

```bash
python -m pip install "triton==3.5.1"
```

Do not install the Triton extra on native Windows. The Torch backend uses CUDA
when a CUDA device is selected; it is not limited to CPU execution. Choose a
CUDA build compatible with your NVIDIA driver. Do not mix unrelated releases of
torch, torchvision and torchaudio.

The validation machines use the following configurations:

| Platform | GPU | Python | PyTorch / vision / audio | Triton |
|---|---|---|---|---|
| Native Windows | RTX 4070 Ti SUPER | 3.12 | 2.9.1 / 0.24.1 / 2.9.1, CUDA 12.6 | Not installed; Torch backend |
| Linux | RTX PRO 6000 Blackwell | 3.13 | 2.11.0 / 0.26.0 / 2.11.0, CUDA 13.0 | 3.6.0 |

For the validated Blackwell configuration, install the matching set instead:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ".[vision,speech,llm,yolo]" "triton==3.6.0"
```

The supported dependency ranges are in `pyproject.toml`. Optional packages are
grouped into `vision`, `speech`, `llm`, `yolo` and `triton` extras. Install only
the extras needed by your examples.

| What you want to run | Extra |
|---|---|
| Matrix multiplication | Base package: `python -m pip install -e .` |
| MNIST, CIFAR, ImageNet, DeiT | `vision` |
| GRU speech | `speech` |
| YOLO VOC | `yolo` |
| Qwen3 | `llm` |
| Optional Linux CUDA kernels | `triton`, matched to the installed PyTorch |

On Windows PowerShell you can avoid activation-policy issues by calling the
environment's Python directly, for example
`.\.venv\Scripts\python.exe -m pip install -e ".[yolo]"`.

### First Run: No Dataset Needed

From the repository root:

```bash
python examples/01_matrix_multiplication.py --backend torch --device cpu
python examples/01_matrix_multiplication.py --backend torch --device cuda:0
```

The first command works without CUDA. Run the second only with a CUDA-enabled
PyTorch build and an available NVIDIA GPU. On Linux with Triton installed:

```bash
python examples/01_matrix_multiplication.py --backend triton --device cuda:0
```

The JSON summary should report the backend you selected, nonzero call counts,
and a finite error relative to FP32. **A nonzero FP32 error is expected** from
operand quantization and ADC conversion, even with all variation set to zero.
With `auto`, inspect the reported fallback reason rather than assuming Triton ran.

## Quick Start

```python
import torch
from memintelli import SimulationEngine, convert_model

engine = SimulationEngine(
    backend="auto",       # auto, triton, torch
    mode="speed",         # speed or accurate
    device="cuda:0",      # or cpu
    activation_bits=6,
    weight_bits=6,
    adc_bits=6,
    adc_clip=True,
    weight_paral_size=(64, 64),
)

model = torch.nn.Sequential(
    torch.nn.Linear(128, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 10),
).to(engine.device).eval()
model = convert_model(model, engine)

with torch.no_grad():
    output = model(torch.randn(4, 128, device=engine.device))
print(engine.describe())
```

For direct matrix multiplication:

```python
weight = engine.map_weight(torch.randn(128, 64, device=engine.device))
output = engine.matmul(torch.randn(4, 128, device=engine.device), weight)
```

Weights use `[input_features, output_features]` order in `map_weight`; PyTorch
Linear weights use the transpose. `convert_model` handles this automatically.
Move a model to its execution device **before** conversion.

### Backend and Numerical Mode

| Option | Behavior |
|---|---|
| `backend="auto"` | Try Triton; visibly fall back to Torch if unavailable or compilation/execution fails. |
| `backend="triton"` | Require Triton. An error is raised instead of silently changing backend. |
| `backend="torch"` | Use PyTorch without importing Triton kernels. Supports CPU and CUDA. |
| `mode="speed"` | BF16 conductance/voltage rounding and BF16-rounded current; FP32 ADC arithmetic and reconstruction. Default. |
| `mode="accurate"` | FP32 conductance, voltage and current; FP32 reconstruction. |

Both backends share the same device samples, quantization and ADC definitions.
They do not switch noise models on fallback. Invalid inputs and CUDA
out-of-memory/device faults are not hidden by a fallback. Backend floating-point
reduction differences can affect values near ADC thresholds; bitwise equality
for every possible input is not promised.

Increasing operand, cell or ADC precision does not remove BF16 rounding in
`speed` mode. Use `accurate` when studying many closely spaced conductance
states or fine ADC thresholds, and compare the two modes on the same inputs.

`mode` is a numerical execution setting, not an alternative array architecture.
The new API uses signed integer bit slicing. Historical `DPETensor` and
`SlicedData` APIs remain available for existing scripts, including their
original BFP functionality; their numeric `mode` options are separate from
this API.

## Array Size and Quantization Granularity

`paral_size` specifies **where array computation and ADC conversion happen**.
`quant_gran` specifies **which values share a quantization scale**.
They are not interchangeable.

For `X[N,K] @ W[K,M]`:

- `weight_paral_size=(R,C)` divides weights into physical arrays with R rows and C columns.
- `input_paral_size` is derived as `(1,R)`. Input block width must equal array row count.
- `weight_quant_gran=(Qr,Qc)` defines a weight scale shared by Qr-by-Qc values.
- `input_quant_gran=(1,Qk)` defines an input scale shared by Qk features within one sample.
- Input feature groups and weight row groups must span complete array rows.
- Weight column groups can be smaller than the array, including one independent scale per column.
- Incomplete boundary blocks are zero padded. Final outputs are cropped to their original shapes.

Example:

```python
engine = SimulationEngine(
    weight_paral_size=(64, 64),
    weight_quant_gran=(128, 128),
    input_quant_gran=(1, 128),
)
```

Here a 128-by-128 weight quantization group shares one scale but uses **four
64-by-64 array blocks per weight slice**. Each block still performs its own ADC
conversion. Enlarging the scale group does not enlarge the physical array.

For column-wise quantization within each 64-row block:

```python
engine = SimulationEngine(
    weight_paral_size=(64, 64),
    weight_quant_gran=(64, 1),
)
```

Each array now has 64 weight scales, one per column. Weights are quantized with
their own column scale; after ADC and bit reconstruction, each column is
rescaled independently. The array remains 64-by-64. `(128,1)` shares each
column's scale across two 64-row blocks. Column groups need not divide the
array width, and boundary groups use only the available values.

Weight row groups smaller than the physical row count are rejected: their
different scales cannot be recovered from one already-summed column current.
Legacy APIs reject unsupported explicit granularities instead of silently
rounding them up. Use `SimulationEngine` for sub-array column granularity.

For a K-by-M matrix with S weight slices, allocated physical-array count is
`ceil(K/R) * ceil(M/C) * S`. Input slices require additional read operations,
not additional copies of the weight arrays.

## Bit Slicing and ADC

The default A6W6 profile uses six one-bit slices for activations and weights.
Slice lists are written from most significant to least significant; the first
slice is the one-bit sign slice. Their sums must match the operand bit widths.
Signed reconstruction uses two's-complement significance.

Multi-bit cells and multi-bit input slices are supported:

```python
engine = SimulationEngine(
    activation_bits=5, weight_bits=5,
    input_slice=(1, 2, 2), weight_slice=(1, 2, 2),
    dac_bits=2, g_level=4, adc_bits=9,
    weight_paral_size=(32, 32),
    adc_clip=True,
)
```

Each input-slice/weight-slice pair has its **own ADC conversion before signed
reconstruction**. A two-bit input slice and two-bit weight slice on 32 rows
have a theoretical unsigned partial-sum maximum of `32 * 3 * 3 = 288`.
This is the maximum of that slice pair, not of the complete signed matrix product.

### `adc_clip=True`

Use a power-of-two ADC step in partial-sum units, rather than stretching
the theoretical maximum onto every available ADC code.

For array row count R, input-slice width a, weight-slice width w, and B ADC bits:

```text
S = R * (2**a - 1) * (2**w - 1)
E = max(1, ceil(log2(S)))
step = 2**(E - B)
limit = min(S, 2**E - 1)
highest_code = min(2**B - 1, floor(limit / step))
code = clamp(round(partial_sum / step), 0, highest_code)
reconstructed_partial_sum = code * step
```

Rounding uses round-to-nearest, ties-to-even. The current is first converted to
partial-sum units using the configured conductance range and read voltage.
In `accurate` mode, to prevent FP32 reduction differences from flipping a
half-code tie, the portable backends snap values to the nearest half-integer only
within `min(4 * float32_epsilon * max(abs(code_value), 1), 1e-4)` ADC codes.
The same rule applies to both ADC settings. This is a numerical tie tolerance,
not a device-noise compensation or reconstruction from an ideal answer.

| Array / slice pair | ADC | Step | Highest used code | Reconstructed maximum |
|---|---:|---:|---:|---:|
| 64 rows, 1-bit x 1-bit | 5-bit | 2 | 31 | 62 |
| 64 rows, 1-bit x 1-bit | 6-bit | 1 | 63 | 63 |
| 64 rows, 1-bit x 1-bit | 7-bit | 0.5 | 126 | 63 |
| 32 rows, 2-bit x 2-bit | 9-bit | 1 | 288 | 288 |

The 7-bit example deliberately leaves code 127 unused. Code 126 is reconstructed
as `126 * 0.5`, not by multiplying by `63/127`.

### `adc_clip=False`

Use conventional full-range scaling:

```text
code = clamp(round(partial_sum / S * (2**B - 1)), 0, 2**B - 1)
reconstructed_partial_sum = code / (2**B - 1) * S
```

For S=64 and B=6, the step is `64/63`; integer partial sums therefore need not
remain integers after reconstruction. Both settings saturate out-of-range
signals; `False` disables the integer-aligned rule, not the ADC range limit.

## Device Variation and Drift

All conductances are in siemens; `vread` is in volts. Variations multiply the
absolute device conductance before subtraction of the nominal low-state
reference:

```text
G = G_nominal * exp(write_sigma[state] * z_write)
G = G * max(drift_time / drift_reference_time, 1)**(-nu[state])
G = G * exp(read_sigma[state] * z_read)
```

`z_write` and `z_read` are standard-normal samples. Sigma is a **log-domain
standard deviation**, not a direct percentage error. The model does not subtract
`sigma**2/2`, so the mean multiplier is `exp(sigma**2/2)`.

Scalar, list and complete state-index dictionaries are accepted:

```python
engine = SimulationEngine(
    g_level=4,
    write_variation={0: 0.02, 1: 0.03, 2: 0.04, 3: 0.05},
    read_variation=[0.01, 0.012, 0.015, 0.02],
    drift_coefficient=[0.01, 0.015, 0.02, 0.025],
    drift_time=10000,
    drift_reference_time=1,
    seed=42,
)
```

- State indices run from the lowest to the highest **nominal** conductance.
- The same state distributions apply to all arrays, with distinct per-cell samples.
- Write samples stay fixed across batches, time steps and software chunks.
- Read samples change on each mapped-layer call. Within that call they are
  shared across batch rows and input slices.
- In the GRU adapter, both input and recurrent projections are read separately
  at every time step. Weight arrays are reused.
- `reset_read_sequence()` replays reads without changing programmed weights.
- `program_epoch` selects a different programming realization. Training
  reprograms changed weights when they are next used.
- Drift uses the explicitly supplied retention time, not wall-clock runtime
  or the index of a GRU time step. Zero time disables drift.

The per-state model is a configurable statistical device model. No unpublished
device measurements or project-specific calibration files are bundled.

## Parameters

| Parameter | Default | Meaning |
|---|---|---|
| `backend` | `"auto"` | Backend selection described above. |
| `mode` | `"speed"` | Internal numerical precision. |
| `device` | CUDA if available, otherwise CPU | Execution device. |
| `activation_bits`, `weight_bits` | `6`, `6` | Signed operand widths, 2 through 16. |
| `input_slice`, `weight_slice` | one-bit slices | Sign-first slice widths; each width is at most 8. |
| `weight_paral_size` | `(64,64)` | Physical array geometry. |
| `input_quant_gran` | `(1,array_rows)` | Per-sample activation scale group. |
| `weight_quant_gran` | array size | Weight scale group. |
| `adc_bits`, `dac_bits` | `6`, `1` | Converter bit widths, 1 through 16. |
| `adc_clip` | `True` | Integer-aligned ADC rule. |
| `HGS`, `LGS` | `1e-5`, `1e-7` | High/low nominal conductance. Must satisfy `0 < LGS < HGS`. |
| `g_level` | `16` | 2 through 256 nominal conductance levels. Must cover the largest weight slice. |
| `vread` | `0.2` | Maximum input read voltage. |
| `write_variation`, `read_variation` | `0` | Scalar or per-state log-domain sigma. |
| `drift_coefficient` | `0` | Scalar or per-state drift exponent. |
| `drift_time` | `0` | Retention time; zero disables drift. |
| `drift_reference_time` | `1` | Positive reference time, in the same unit as retention time. |
| `seed` | `42` | Device/read seed, integer from 0 through `2**32-1`. |
| `program_epoch` | `0` | Nonnegative programming-round identifier. |
| `input_chunk_rows` | `256` | Maximum sample/token rows processed together. |
| `output_chunk_tiles` | `8` | Maximum output-array columns processed together. |
| `torch_fuse_adc` | `True` | Enable equivalent CUDA ADC/reconstruction optimizations in the Torch backend. |

Chunk sizes control temporary memory, not physical array size or quantization
granularity. Device samples are addressed by logical position so chunk changes
do not select new write/read samples. Keep numerical and device parameters
fixed after mapping; software chunk sizes may be changed. Construct a new
engine for a different device model.

### Valid Configuration Checklist

- Operand widths are 2-16 bits because a sign bit is included. Slice lists are
  nonempty, start with `1`, and sum to the corresponding width.
- Each slice is 1-8 bits. `dac_bits >= max(input_slice)` and
  `g_level >= 2**max(weight_slice)`.
- Array dimensions and chunk sizes are positive integers.
- Input scale groups are `(1,Qk)` with Qk a multiple of the physical row count.
  Weight scale groups are `(Qr,Qc)` with Qr a multiple of the physical row count;
  Qc may be 1. Array shape and scale group are separate settings.
- High/low conductances and voltage are positive and finite, `LGS < HGS`,
  and their conductance step and ADC reference must fit normal FP32 arithmetic.
- Variation and drift coefficients are nonnegative and finite. State lists have
  exactly `g_level` entries; dictionaries cover every state once.
- Retention time is nonnegative, reference time is positive, and programming
  epoch is a nonnegative integer.
- Inputs/weights are finite real dense tensors, with no empty dimension.
  A mapped weight belongs to the engine that created it.

Invalid configurations raise an error naming the parameter; they are not
silently rounded into a different simulation. Valid configurations can still
exceed available memory. Smaller software chunks reduce temporary memory
without changing the array geometry.

## Check Acceleration and Runtime

After a forward call, `engine.describe()` tells you what actually ran:

| Field | Check |
|---|---|
| `device` | `cuda:0` for GPU execution, not `cpu` |
| `backend_used` | `triton` or `torch`; `auto` is only the requested policy |
| `mode` | `speed` uses BF16 current computation; `accurate` uses FP32 |
| `calls.triton`, `calls.torch` | Number of executed backend calls |
| `fallback_reason` | Why automatic selection changed to Torch, if applicable |
| `torch_adc_fusion` | `active` for fused CUDA ADC, `lookup` for an exact BF16 lookup, `not_used`, `disabled`, or `unavailable` |

Native Windows uses **Torch CUDA**, not Triton, in the documented installation.
The speed path calls BF16 batched GPU matrix multiplication. Optional PyTorch
NVRTC kernels fuse activation quantization/bit slicing and voltage preparation.
For binary slices, DAC1, clipped ADC of up to eight bits and no array-reference
correction, input and weight slices are grouped into larger GPU operations.
One kernel performs every per-slice ADC, signed accumulation and output scaling,
then PyTorch sums the array-row blocks. This path is used only within a checked
FP32 exact-integer accumulation bound; other cases retain the general path.
Intermediate BF16 current chunks target 32 MiB to limit memory traffic.

The optional grouped kernels use PyTorch's internal CUDA compilation API, checked
at runtime. `torch_cuda_fusion` reports `active`, `not_used`, `disabled` or
`unavailable`; `torch_cuda_fusion_error` records a compilation failure. Missing
runtime support warns and retains the previous portable implementation.
Up to seven weight slices can still use Jiterator ADC fusion, reported separately
by `torch_adc_fusion` and `torch_adc_fusion_error`. Neither mechanism requires
Triton or a separately installed `nvcc` compiler.

If both compilation paths are unavailable, execution continues in ordinary Torch.
Where safe, this
uses an exact table of the ADC function over BF16 current encodings; otherwise
it uses the ordinary arithmetic path. The lookup uses the simulated current,
not an ideal output or test label. FP32 mode, multi-bit cells, and array-reference
correction retain the general current/ADC path. Compilation failures do not
resample noise; out-of-memory and fatal CUDA errors are not silently hidden.

For a direct unfused comparison, set `torch_fuse_adc=False` or use
`--no-torch-fuse-adc`. This disables the Torch preparation and ADC fusion kernels.
It is a software optimization switch, not a different
device model. It does not change ADC saturation, random samples, physical-array
geometry or quantization granularity.

The Triton backend uses fused activation quantization/bit slicing and a tuned
64-row current kernel for common speed-mode shapes. Input and weight bit slices
still receive separate ADC conversions; tuning GPU tiles never enlarges the
physical array. Large quantization groups retain the portable input slicer.

`input_chunk_rows` groups independent sample/token or unfolded image-patch rows.
Increasing it reduces small calls but increases temporary memory.
`output_chunk_tiles` groups output-array blocks. Neither changes `paral_size`,
ADC width or quantization granularity. Start with a short run, try larger chunks,
and compare outputs before a full evaluation. Do not change physical arrays
or quantization just to report a faster runtime.

The YOLO example reports first-image forward time, subsequent mean forward
time, overall evaluation time and peak allocated GPU memory. The first image
includes lazy weight mapping and possible Triton or Torch CUDA compilation. CUDA is
synchronized for timing. A short-run projection is an estimate, not a measured
full-dataset runtime or physical-chip latency.

## Troubleshooting

| Symptom | What to check |
|---|---|
| `No module named triton` | Expected for Windows `auto`; use `--backend torch`. For strict Triton, use the documented Linux installation. |
| `CUDA was requested but is unavailable` | Check `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` in the environment running the example. |
| CUDA out of memory | Reduce batch size, `--input-chunk-rows`, then `--output-chunk-tiles`; keep model and physical parameters unchanged. |
| A `quant_gran` error | Check row alignment; `(64,1)` is valid for a `(64,64)` weight array, but `(32,1)` is not. |
| State-variation length mismatch | Match the list/dictionary to `g_level`, not to the number of weight bits. |
| Checkpoint checksum mismatch | Remove only the named corrupt cached checkpoint and rerun; do not bypass verification. |
| Dataset not found | `--checkpoint` supplies weights, not data. Follow the directory layout in the examples README. |
| Slow first image | Mapping or kernel compilation can be included; compare later images separately. |
| NVRTC/ADC fusion warning | Check the matching PyTorch CUDA runtime libraries. Inference still works; `--no-torch-fuse-adc` explicitly selects the unfused path. |
| Quantized accuracy below FP32 | Compare the digital model, no-noise simulation, then each variation condition. A6W6 is not an FP32-equivalent setting. |

## Neural Network Support

`convert_model` maps Linear, Conv2d (including grouped/depthwise convolution),
and unidirectional zero-dropout GRU modules. Explicit numeric Conv2d padding is
required. GRU inputs must be dense three-dimensional sequences.

Batch normalization, pooling, nonlinear activations, recurrent state updates,
embeddings, attention softmax and dynamic attention matrix products remain
digital. Bias is added digitally. Qwen3 examples map every Linear, including
the output projection; this is not an attention-matrix hardware simulator.

The training examples use simulated forward results with a digital
straight-through gradient. This is software hardware-aware training, not a
physical on-chip programming/update model.

## Examples and Checkpoints

See [examples/README.md](examples/README.md) for runnable commands, datasets,
model choices and backend flags. Examples do not contain stored results or
result screenshots. `--output` writes a summary only when requested.

The [v2.0.0 model release](https://github.com/HUST-ISMD-Odyssey/MemIntelli/releases/tag/v2.0.0)
contains:

- `gru_speech_commands.pt`: two-layer 128-unit GRU, classification weights,
  35-class order, feature settings and training-derived normalization.
- `yolov3_voc.pt`: VOC detection weights and model configuration.
- `yolov5s_voc.pt`: the smaller VOC detection alternative.
- `checksums.json`: SHA-256 values and file sizes.

The GRU/YOLO examples use these URLs by default and verify the downloaded
checkpoint hashes. They load the new assets with `weights_only=True`.
Custom checkpoints and YOLO model configurations must come from trusted sources.

### Sources and Licensing

MemIntelli's license is in [license.txt](license.txt). Historical source
attribution is retained. The GRU was trained using Google's Speech Commands
v0.02 data; raw audio is not redistributed here.

YOLO weights originate from [zjykzj/YOLOv5](https://github.com/zjykzj/YOLOv5),
release `v1.0`. They are repackaged as state dictionaries without retraining.
The example downloads the upstream implementation at a pinned revision into
the user's cache, or accepts a local checkout. Upstream contains its own license
and file-level notices, including Ultralytics GPL notices; those rights and
obligations are not replaced by MemIntelli's license.

CIFAR model sources are identified by the URLs in `memintelli/NN_models`.
ImageNet models use torchvision or the published DeiT implementation.
Qwen3-0.6B is downloaded from its original model repository. Dataset downloads,
third-party source, checkpoints and generated results are not committed to
this source branch.
