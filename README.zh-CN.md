# MemIntelli 中文入门

[完整参数说明](README.md) | [各示例与数据集说明](examples/README.md)

MemIntelli 用于模拟神经网络在忆阻器阵列上的计算：量化、权重映射、
阵列电流、ADC 转换、器件误差和输出重构。
它是仿真器，不是让普通神经网络推理变快的工具。

## 1. 先跑通一个不需要数据集的例子

建议创建独立环境，不修改已有项目的 Python 环境。
下面以 Windows、Python 3.12 和 NVIDIA GPU 为例：

```powershell
git clone --branch v2 https://github.com/HUST-ISMD-Odyssey/MemIntelli.git
cd MemIntelli
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe examples/01_matrix_multiplication.py --backend torch --device cuda:0
```

这些命令直接使用环境内的 Python，不需要修改 PowerShell 执行策略。
如果没有安装 Python 3.12，先安装该版本，或用已有的受支持 Python 创建环境。
如果没有 NVIDIA GPU，按英文 README 安装 CPU 版 PyTorch，最后一条命令改用
`--device cpu`。Linux 环境及对应 Triton 版本也见英文 README。

输出中重点看：

| 字段 | 含义 |
|---|---|
| `backend_used` | 实际执行的是 `torch` 还是 `triton` |
| `device` | `cuda:0` 表示使用 GPU；`cpu` 表示 CPU |
| `mode` | 默认 `speed`，使用 BF16 电流计算 |
| `calls` | 对应后端的实际调用次数应大于零 |
| `fallback_reason` | 自动回退的原因，没有回退时为空 |

矩阵例子会与 FP32 结果比较。即使没有器件噪声，量化和 ADC 也可能产生误差，
因此误差不为零不代表程序出错。

## 2. A6W6、阵列大小和量化粒度分别是什么

- **A6W6**：输入激活和权重分别用 6 位有符号整数表示。
- **`weight_paral_size=(64,64)`**：每个物理阵列有 64 行、64 列。
- **`weight_quant_gran=(64,64)`**：每个 64 行、64 列的权重块共享一个量化尺度。
- **`weight_quant_gran=(64,1)`**：每个 64 行块内，各列独立量化；同一阵列有 64 个尺度。
- **`weight_quant_gran=(128,1)`**：每列的尺度跨两个 64 行阵列块共享。
- **`input_quant_gran=(1,64)`**：每个输入样本的每 64 个特征共享一个尺度。

例如：

```python
import torch
from memintelli import SimulationEngine

engine = SimulationEngine(
    backend="torch",
    device="cuda:0",
    activation_bits=6,
    weight_bits=6,
    weight_paral_size=(64, 64),
    weight_quant_gran=(64, 1),
    input_quant_gran=(1, 64),
    adc_bits=6,
    adc_clip=True,
)
weight = torch.randn(128, 64, device=engine.device)
inputs = torch.randn(4, 128, device=engine.device)
mapped_weight = engine.map_weight(weight)
outputs = engine.matmul(inputs, mapped_weight)
print(outputs.shape)  # torch.Size([4, 64])
print(engine.describe())
```

直接矩阵接口的权重形状是 `[输入特征数, 输出特征数]`。
PyTorch `Linear.weight` 的顺序相反；使用 `convert_model` 时会自动处理。
同一个映射权重需要交给创建它的引擎使用。

量化分组不改变阵列尺寸。权重的行分组必须是阵列行数的整数倍，
列分组可以小于阵列列数。程序不会把 `64×1` 自动扩大为 `64×64`。

## 3. ADC 位宽不等于输入或权重位宽

默认 A6W6 拆成六个一位输入切片和六个一位权重切片。
仿真需要处理这些位片组合及各自的 ADC 转换，然后重构有符号输出，
不是直接调用一次“6 位 GPU 卷积”。

`adc_clip=True` 使用与部分和对齐的二进制步长：

| 阵列行数与物理位片 | ADC | 每码对应的部分和 | 使用的最大码 | 重构上限 |
|---|---|---|---|---|
| 64 行，一位输入、一位权重 | 5 位 | 2 | 31 | 62 |
| 64 行，一位输入、一位权重 | 6 位 | 1 | 63 | 63 |
| 64 行，一位输入、一位权重 | 7 位 | 0.5 | 126 | 63 |
| 32 行，两位输入、两位权重 | 9 位 | 1 | 288 | 288 |

7 位的例子使用 `126×0.5=63`，不做 `127/63` 缩放。
`adc_clip=False` 使用传统满量程缩放，仍然会限制 ADC 范围，并不是取消饱和。
FP32 模式在恰好处于 ADC 半码边界时使用很小的数值容差，
避免两种后端的浮点舍入差异改变判码；这不是器件噪声补偿，公式见英文说明。

多位切片需要同时调整 DAC 和器件状态数。例如：

```python
engine = SimulationEngine(
    backend="torch", device="cuda:0",
    activation_bits=5, weight_bits=5,
    input_slice=(1, 2, 2), weight_slice=(1, 2, 2),
    dac_bits=2, g_level=4, adc_bits=9,
    weight_paral_size=(32, 32),
)
```

切片从符号位开始，位宽之和必须等于总位宽；首个切片必须是 `1`。
`g_level` 是器件可用的电导状态数，不是整个权重的位宽。
默认一位权重切片只使用最低、最高两个状态。

## 4. 再运行一个网络

只安装需要的扩展：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[vision]"
.\.venv\Scripts\python.exe examples/07_resnet_cifar_inference.py --backend torch --download --limit 2
```

这会获取该示例的模型和 CIFAR-10 数据，首次下载不计作仿真速度。
先运行两个样本确认环境和路径，再移除 `--limit` 完整评测。

### 语音 GRU

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[speech]"
.\.venv\Scripts\python.exe examples/11_gru_speech.py --backend torch --wav D:\data\example.wav
```

输入是单声道、16 kHz、不超过一秒的 WAV。默认模型包含特征提取和归一化参数。
完整测试使用 Speech Commands v0.02 官方测试划分，详见示例 README。

### YOLOv3 / VOC2007

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[yolo]"
.\.venv\Scripts\python.exe examples/12_yolov3_voc_inference.py --backend torch --data-root D:\data\VOCdevkit\VOC2007 --limit 2
```

VOC2007 测试目录必须包含：

```text
VOC2007/
  JPEGImages/
  Annotations/
  ImageSets/Main/test.txt
```

权重会从本仓库 Release 下载；VOC 数据本身需要另行准备。
`--download` 不会自动下载 VOC。默认评测分辨率为 640×640，每次处理一张图。
`--batch-size` 不改变这个 YOLO 示例的逐图流程。
`--model yolov5s` 可切换为另一份公开权重。

程序输出 VOC2007 的 11 点插值 `mAP@0.5`、仿真配置和计时。
使用 `--limit` 得到的是子集结果，不能当成完整测试集精度。

## 5. 正确使用 GPU 和软件分块

- `backend="auto"`：先尝试 Triton，失败后明确警告并回退到 Torch。
- `backend="triton"`：必须使用 Triton，失败直接报错。
- `backend="torch"`：直接使用 PyTorch；选择 CUDA 时仍在 GPU 上计算。
- `mode="speed"`：BF16 电压、电导和电流计算，ADC 与重构使用 FP32。
- `mode="accurate"`：使用 FP32 计算，不等于取消量化或器件误差。

本文的原生 Windows 环境没有安装 Triton，用的是 **Torch CUDA 的 BF16 批量矩阵乘法**。
`speed` 不是省略器件仿真，也不是跳过卷积层。
增加输入、权重或 ADC 位宽并不会取消 BF16 舍入；研究很多相邻电导状态或
很细的 ADC 分辨率时，应同时用 `accurate` 检查数值精度。

Torch 后端默认启用 `torch_fuse_adc=True`。对常用的一位切片、DAC1、
`adc_clip=True`、ADC不超过8位、无阵列参考校正的配置，会合并输入和权重位片的矩阵乘法，
并把各位片的ADC、带符号累加和输出缩放合成一个CUDA内核。
输入量化、位切片和电压数据转换也使用融合内核，减少中间数据读写。
带噪声时，器件地址、随机数、分电导态的写入误差和读噪声、漂移乘法及电导转换
也使用融合内核。同一次逻辑读取中的输入分块可以复用电导；下一次读取重新生成
读噪声，不跨GRU时间步复用。写入误差仍按器件位置和编程轮次固定。
程序检查FP32整数累加范围，超出该范围则保留通用路径。
Windows 不需要安装 Triton，也不需要单独安装 `nvcc`。

查看 `torch_cuda_fusion`：`active` 表示新的CUDA融合内核已执行，
`unavailable` 表示当前PyTorch运行时不支持或编译失败，原因见 `torch_cuda_fusion_error`。
该优化使用PyTorch内部编译接口，程序会检查是否可用；不可用时警告并保留原计算路径，
其中仍可能使用上一种Jiterator ADC融合。
查看 `torch_adc_fusion`：`active` 表示融合内核已执行，`lookup` 表示使用
等价的 BF16 电流查表；`not_used` 表示本次未使用，`disabled` 表示手动关闭，
`unavailable` 表示编译不可用，具体原因见 `torch_adc_fusion_error`。
编译不可用时会警告并继续使用普通 Torch 路径。查表只预先计算 ADC 函数，
输入仍是实际仿真电流，不是用理想答案替代阵列计算。

用 `--no-torch-fuse-adc`，或设置 `torch_fuse_adc=False`，可关闭Torch输入准备及ADC融合，
与未融合路径对照。多位器件、FP32模式和阵列参考校正仍使用通用电流及ADC计算路径。
编译失败回退不改变噪声采样；显存不足或CUDA执行错误不会被静默忽略。
Triton 后端则融合输入量化、位切片，并对常用64行配置调整GPU计算块。
这些优化都保留每个位片、每个物理阵列的ADC，不改变噪声或量化粒度。

默认使用 `chunk_policy="auto"`、`workspace_mb=512`，即512 MiB临时工作区预算。
程序根据当前层的输入数量、权重形状和位片数，联合选择输入分块和权重输出方向分块。
这是一套按形状估算的调度策略，不做训练、不使用标签，也不执行测速搜索；
不能保证在每张显卡上都是最快配置。

```powershell
.\.venv\Scripts\python.exe examples/12_yolov3_voc_inference.py --backend torch --data-root D:\data\VOCdevkit\VOC2007 --limit 10 --workspace-mb 512
```

预算包括电导、电压和中间电流等临时数据，预留四分之一用于同一次读取的电导复用。
它不是GPU总显存上限，不包含模型、映射索引、卷积展开、KV缓存、首次映射等占用。
预算过小可能产生大量小调用；预算增大也不一定更快。显存紧张时先减小批量和
`--workspace-mb`，不要直接改动物理阵列。最小计算块仍可能超出很小的预算，
实际选定的分块及估计占用可在 `engine.describe()["chunk_plans"]` 中查看。

自动调度用于满足融合条件的Torch CUDA一位切片、截断ADC、speed路径。
Triton、CPU、accurate模式、多位切片、阵列参考校正及融合不可用时使用手动分块。
若需要明确控制分块，设置 `chunk_policy="manual"`，或：

```powershell
.\.venv\Scripts\python.exe examples/12_yolov3_voc_inference.py --backend torch --data-root D:\data\VOCdevkit\VOC2007 --limit 10 --chunk-policy manual --input-chunk-rows 4096 --output-chunk-tiles 8
```

`input_chunk_rows` 是一次处理的样本、词元或卷积展开输入行数；
`output_chunk_tiles` 是一次处理的权重阵列列块数。手动模式下，YOLO默认输入上限
为16384，通用引擎为256。两者都不改变64×64物理阵列、A6W6、量化粒度或噪声地址。
分块变化可能改变浮点求和顺序，产生末位误差；严格复现时固定手动分块并比较输出。

第一次推理可能包含权重映射、Triton 或 Torch CUDA 内核编译，应与后续图片分开计时。
程序报告的 GPU 仿真耗时不是芯片延迟。用少量图片推算完整测试集时长，
需要明确标为估计值。

## 6. 最后再加入器件误差

先跑数字基准，再跑无器件噪声的仿真，最后逐项增加写误差、读误差和漂移。
不要第一次就同时改变量化、阵列、噪声和模型。

```python
engine = SimulationEngine(
    backend="torch", device="cuda:0",
    g_level=4,
    write_variation=[0.01, 0.02, 0.03, 0.04],
    read_variation=[0.005, 0.01, 0.015, 0.02],
    drift_coefficient=[0.01, 0.01, 0.02, 0.02],
    drift_time=10000,
    drift_reference_time=1,
    seed=42,
)
```

这里的 variation 是**对数域标准差**，不是直接的百分比误差。
写误差在映射后固定；读误差在每次逻辑层调用时更新，
同一次调用内由批次和输入位片共享。GRU 的输入、循环线性变换在每个时间步分别读取，
但不会复制一套新权重。漂移使用指定保持时间，不随程序运行时间自动增加。
具体公式与各参数默认值以英文 README 为准。

## 7. 常见问题

| 问题 | 处理 |
|---|---|
| Windows 提示没有 Triton | 使用 `--backend torch`；`auto` 的回退警告是预期行为 |
| 明明有 GPU 却显示 CPU | 检查运行示例的 Python 环境、PyTorch CUDA 构建和 `--device` |
| `64×1` 量化粒度不生效 | 使用 `SimulationEngine`；旧接口不支持的粒度会明确报错 |
| variation 配置长度错误 | 列表长度必须等于 `g_level`，字典必须覆盖全部状态 |
| `input_slice=[]` 报错 | 空切片不合法；省略参数使用默认值 |
| 显存不足 | 先减批量和软件分块，不改变物理仿真配置 |
| 下载了权重仍找不到数据 | checkpoint 不包含测试数据，检查 `--data-root` |
| 无噪声准确率也低于 FP32 | 量化和 ADC 本身有误差，先与 `--digital` 比较 |

程序会拒绝非法参数，而不是保证任意参数组合都能执行。
正常配置也受显存、数据文件和设备支持情况限制。边界验证覆盖了位宽上下界、
多位切片、非整齐阵列、逐列量化、零矩阵、分块、读写状态及两后端对照。
