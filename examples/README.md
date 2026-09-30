# Examples

[中文入门指南](../README.zh-CN.md) | [Complete parameter reference](../README.md)

Run commands from the repository root after installing the required extras.
All examples default to `--backend auto --mode speed --adc-clip`.
Use `--backend torch` for native Windows, or `--backend triton` to require
Triton without fallback.

Torch CUDA enables input preparation and grouped ADC fusion by default for
supported speed-mode settings. The grouped kernel combines per-slice ADC,
signed accumulation and output scaling without changing the physical arrays.
Use `--no-torch-fuse-adc` for an unfused comparison. Check
`simulation.torch_cuda_fusion` and `simulation.torch_cuda_fusion_error` for the
new grouped path; `simulation.torch_adc_fusion` reports ADC fusion including
the older Jiterator fallback. Compilation failures warn and keep portable Torch
available. First-call compilation is separate from steady inference.
This does not require Triton on Windows.

## Order

| No. | Script | Task / dependencies |
|---|---|---|
| 01 | `01_matrix_multiplication.py` | Matrix multiplication; base package |
| 02 | `02_MLP_inference.py` | MNIST MLP; vision |
| 03 | `03_MLP_hardware_aware_training.py` | MNIST hardware-aware training; vision |
| 04 | `04_mlp_hardware_aware_training_ddp.py` | Distributed MNIST training; vision |
| 05 | `05_vgg_cifar_inference.py` | VGG16-BN / CIFAR10; vision |
| 06 | `06_vgg16bn_cifar100_finetune.py` | VGG16-BN / CIFAR100 fine-tuning; vision |
| 07 | `07_resnet_cifar_inference.py` | ResNet18 / CIFAR10; vision |
| 08 | `08_vgg_imagenet_inference.py` | VGG16-BN / ImageNet; vision |
| 09 | `09_resnet_imagenet_inference.py` | ResNet18 / ImageNet; vision |
| 10 | `10_mobilenetv2_imagenet_inference.py` | MobileNetV2 / ImageNet; vision |
| 11 | `11_gru_speech.py` | Two-layer GRU / Speech Commands v0.02; speech |
| 12 | `12_yolov3_voc_inference.py` | YOLOv3 or YOLOv5s / VOC2007; yolo |
| 13 | `13_deit_imagenet_inference.py` | DeiT-Tiny / ImageNet; vision |
| 14 | `14_qwen3_inference.py` | Qwen3-0.6B generation; llm |

The underscore-prefixed files are shared implementation helpers, not separate
examples.

## First Network Run

Start with the matrix example, then one short network evaluation:

```bash
python examples/01_matrix_multiplication.py --backend torch --device cpu
python examples/07_resnet_cifar_inference.py --backend torch --download --limit 2
```

Check `simulation.backend_used`, `simulation.device` and the backend call
counts. Repeat the network command with `--digital` to evaluate the FP32
software model. Remove `--limit` only after the short run works.
The first download retrieves the whole required dataset/checkpoint, even when
only two samples will be evaluated.

## Matrix and Quantization

```bash
python examples/01_matrix_multiplication.py --backend torch
python examples/01_matrix_multiplication.py --backend triton --shape 8 256 128
python examples/01_matrix_multiplication.py --array-size 32 32 --input-quant-gran 1 64 --weight-quant-gran 64 64
python examples/01_matrix_multiplication.py --array-size 64 64 --weight-quant-gran 64 1
python examples/01_matrix_multiplication.py --activation-bits 5 --weight-bits 5 --input-slice 1 2 2 --weight-slice 1 2 2 --dac-bits 2 --g-level 4 --array-size 32 32 --adc-bits 9
```

`--no-adc-clip` selects the older full-range ADC scaling rule. The root README
defines both ADC rules, including unused upper codes.

## MNIST and Training

```bash
python examples/02_MLP_inference.py --download --data-root data --epochs 3
python examples/03_MLP_hardware_aware_training.py --download --data-root data --epochs 3 --read-variation 0.02
torchrun --standalone --nproc_per_node=2 examples/04_mlp_hardware_aware_training_ddp.py --download --data-root data --epochs 3
```

Example 02 trains a digital MLP when `--checkpoint` is omitted, then evaluates
the mapped model. Supply a saved checkpoint to skip training.

Examples 03, 04 and 06 use simulated forward computation and straight-through
digital gradients. `--learning-rate`, `--epochs` and `--save-checkpoint`
configure training. DDP uses separate model replicas per GPU; it does not split
one array simulation across GPUs. For a one-process run, example 04 also works
with plain `python`.

## CIFAR and ImageNet

```bash
python examples/05_vgg_cifar_inference.py --download --data-root data
python examples/06_vgg16bn_cifar100_finetune.py --download --data-root data --epochs 1
python examples/07_resnet_cifar_inference.py --download --data-root data
python examples/08_vgg_imagenet_inference.py --data-root /path/to/imagenet
python examples/09_resnet_imagenet_inference.py --data-root /path/to/imagenet
python examples/10_mobilenetv2_imagenet_inference.py --data-root /path/to/imagenet
python examples/13_deit_imagenet_inference.py --data-root /path/to/imagenet
```

Classification examples download their published weights unless `--checkpoint`
is supplied. ImageNet itself is not downloaded. Provide an ImageFolder layout
`imagenet/val/<class>/<image>` with the standard 1,000-class folder ordering.
You can also pass the `val` directory directly. ImageNet preprocessing is
resize to 256, center crop to 224, and the standard mean/std normalization.

## GRU Speech Commands

```bash
python examples/11_gru_speech.py --data-root data --download
python examples/11_gru_speech.py --data-root /path/to/speech_commands_v0.02 --backend torch
python examples/11_gru_speech.py --wav /path/to/one_second.wav --backend torch
```

The default checkpoint comes from the MemIntelli v2.0.0 release and includes
all normalization parameters. Each mono 16 kHz waveform is padded to one second
and converted to 51 frames of 40-dimensional log-Mel features. Two GRU layers
have 128 hidden units each. The final state feeds a 35-class Linear head.
Full evaluation uses the official `testing_list.txt`; it never fits
normalization on the test split.

Both GRU input and recurrent projections use the array backend at every time
step. Gates and state updates run digitally. There are no new physical copies
of the weights for different time steps.

## YOLO VOC

```bash
python examples/12_yolov3_voc_inference.py --data-root /path/to/VOCdevkit/VOC2007 --model yolov3
python examples/12_yolov3_voc_inference.py --data-root /path/to/VOCdevkit/VOC2007 --model yolov5s --backend torch
```

Provide the extracted VOC2007 **test** set, including `JPEGImages`,
`Annotations` and `ImageSets/Main/test.txt`. The example evaluates one image at
a time at 640-by-640 by default. `--image-size` can select another multiple of
32. `--confidence` defaults to 0.001 and `--nms-iou` to 0.6.
`--download` does not fetch VOC, and `--batch-size` does not change this
example's one-image-at-a-time evaluation.

```text
VOC2007/
  JPEGImages/
  Annotations/
  ImageSets/Main/test.txt
```

Triton speed mode and eligible Torch CUDA inference use `--chunk-policy auto --workspace-mb 512`
to select input and weight chunks jointly. The YOLO manual/fallback input limit
is 16384, compared with 256 in the general engine. To control it explicitly, use
`--chunk-policy manual --input-chunk-rows 4096 --output-chunk-tiles 8`.
Neither policy changes A6W6, physical arrays, quantization groups or noise samples.

For a short timed check, then a complete run:

```bash
python examples/12_yolov3_voc_inference.py --backend torch --data-root /path/to/VOC2007 --limit 10
python examples/12_yolov3_voc_inference.py --backend torch --data-root /path/to/VOC2007 --output voc_summary.json
```

The summary separates initial setup, the first forward call (including lazy
mapping), subsequent mean/p50/p95 forward times and total evaluation wall time.
GPU operations are synchronized for timing. A projection from `--limit 10`
is not a measured full-test-set duration. The reported GPU memory is PyTorch's
peak allocated memory, not total board usage.

Both checkpoints default to this repository's release URLs. The optional YOLO
implementation is downloaded from the pinned zjykzj/YOLOv5 revision. To run
without that download, pass `--yolo-source /path/to/checkout`.

Reported mAP50 uses **VOC2007 11-point AP at IoU 0.5**, ignores detections
matched to difficult objects, and counts duplicate detections as false
positives. This differs from the 101-point integration used by some YOLO
evaluation tools. A limited run averages only classes present in that subset;
it is not a full VOC benchmark.

## Qwen3

```bash
python examples/14_qwen3_inference.py --model Qwen/Qwen3-0.6B --max-new-tokens 16
python examples/14_qwen3_inference.py --model /path/to/Qwen3-0.6B --local-files-only --backend torch
```

The default is Qwen3-0.6B, not a larger model. Linear weights are mapped;
embeddings and dynamic attention operations remain digital. The example uses
greedy decoding with thinking disabled. Reduce `--workspace-mb` if temporary GPU
memory is limited; manual mode also accepts `--input-chunk-rows` and
`--output-chunk-tiles`. Model loading and initial weight mapping have separate
memory requirements.

## Shared Options

Every example exposes `--help`. Important common arguments are:

- `--backend auto|triton|torch`, `--mode speed|accurate`, `--device cpu|cuda:0`.
- `--activation-bits`, `--weight-bits`, `--input-slice`, `--weight-slice`.
- `--array-size ROWS COLS`, `--input-quant-gran ROWS COLS`, `--weight-quant-gran ROWS COLS`.
- `--adc-bits`, `--dac-bits`, `--adc-clip` / `--no-adc-clip`.
- `--hgs`, `--lgs`, `--g-level`, `--vread`.
- `--write-variation`, `--read-variation`, `--variation-json`.
- `--drift-coefficient`, `--drift-time`, `--drift-reference-time`.
- `--seed`, `--program-epoch`.
- `--input-chunk-rows`, `--output-chunk-tiles`.
- `--chunk-policy auto|manual`, `--workspace-mb` (MiB; default 512).
- `--data-root`, `--checkpoint`, `--batch-size`, `--workers`.
- `--limit N` for a shorter dataset evaluation; zero means the complete split.
- `--digital` for floating-point model evaluation without array simulation.
- `--output path.json` to save a summary. Nothing is saved by default.

The matrix example always evaluates the array computation; `--digital` is for
network examples. The Qwen example uses prompt/token limits rather than
dataset/batch arguments.

A variation JSON file can specify a scalar, state list, or complete state
dictionary for `write_variation`, `read_variation`, and `drift_coefficient`.
For a four-state device:

```json
{
  "write_variation": {"0": 0.02, "1": 0.03, "2": 0.04, "3": 0.05},
  "read_variation": [0.01, 0.012, 0.015, 0.02],
  "drift_coefficient": 0.01
}
```

Pass it with `--g-level 4 --variation-json device.json`. Sigma is in the log
domain. See the root README for the exact noise formula and sampling lifetime.
