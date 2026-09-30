"""Dense Conv2d lowering to the v3.1 bit-sliced Triton inference path."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .inference import linear_from_torch
from .pimpy import TritonEngine


class TritonConv2d(nn.Module):
    """Inference-only groups=1 Conv2d with explicit unfold workspace.

    Array reduction uses the same flattened receptive-field order as PyTorch
    unfold and the existing Conv2dMem. Padding is applied before CIM quantization.
    """

    def __init__(self, source, engine, *, streaming=False):
        super().__init__()
        if not isinstance(source, nn.Conv2d) or not isinstance(engine, TritonEngine):
            raise TypeError("Expected torch.nn.Conv2d and TritonEngine")
        if source.groups != 1 or source.padding_mode != "zeros" or isinstance(source.padding, str):
            raise ValueError("Only groups=1, numeric zero-padding Conv2d is supported")
        if source.weight.dtype != torch.bfloat16:
            raise ValueError("TritonConv2d requires BF16 weights")
        self.kernel_size = source.kernel_size
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        dense = nn.Linear(source.in_channels * math.prod(source.kernel_size),
                          source.out_channels, bias=source.bias is not None,
                          device="cpu", dtype=torch.bfloat16)
        with torch.no_grad():
            dense.weight.copy_(source.weight.detach().flatten(1).cpu())
            if source.bias is not None:
                dense.bias.copy_(source.bias.detach().cpu())
        self.linear = linear_from_torch(dense, engine, streaming=streaming, free_weights=True)
        self.eval()

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("Expected NCHW input with matching channels")
        kh, kw = self.kernel_size
        oh = (x.shape[2] + 2 * self.padding[0] - self.dilation[0] * (kh - 1) - 1) // self.stride[0] + 1
        ow = (x.shape[3] + 2 * self.padding[1] - self.dilation[1] * (kw - 1) - 1) // self.stride[1] + 1
        patches = F.unfold(x, self.kernel_size, dilation=self.dilation,
                           padding=self.padding, stride=self.stride).transpose(1, 2)
        result = self.linear(patches)
        return result.transpose(1, 2).reshape(x.shape[0], self.out_channels, oh, ow)


@torch.no_grad()
def replace_cnn_layers(model, engine, *, streaming=False):
    """Replace every dense Conv2d and Linear. Reject grouped conv before editing.

    BatchNorm, activations, pooling and residual additions remain native PyTorch.
    No pretrained weights or datasets are downloaded.
    """
    if not isinstance(engine, TritonEngine):
        raise TypeError("Expected TritonEngine")
    originals = [(name, layer) for name, layer in model.named_modules(remove_duplicate=False)
                 if name and isinstance(layer, (nn.Conv2d, nn.Linear))]
    if not originals:
        raise ValueError("No child Conv2d or Linear modules found")
    for name, layer in originals:
        if layer.weight.dtype != torch.bfloat16:
            raise ValueError(f"{name}: BF16 weights required")
        if isinstance(layer, nn.Conv2d) and (
            layer.groups != 1 or layer.padding_mode != "zeros" or isinstance(layer.padding, str)
        ):
            raise ValueError(f"{name}: grouped convolution or unsupported padding; no layers replaced")
    converted = {}
    for name, layer in originals:
        if id(layer) not in converted:
            converted[id(layer)] = (
                TritonConv2d(layer, engine, streaming=streaming)
                if isinstance(layer, nn.Conv2d) else
                linear_from_torch(layer, engine, streaming=streaming, free_weights=True)
            )
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, converted[id(layer)])
    model.eval()
    return [name for name, _ in originals]
