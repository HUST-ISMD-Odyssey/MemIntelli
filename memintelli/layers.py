"""PyTorch model conversion; nonlinear operations stay on the digital path."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .simulator import SimulationEngine


class _MappedModule(nn.Module):
    def _mapped(self, matrices):
        version = -1 if torch.is_inference(self.weight) else self.weight._version
        if getattr(self, "_weight_version", None) != version:
            with torch.no_grad():
                previous = getattr(self, "_matrices", None)
                if previous is None:
                    self._matrices = [self.engine.map_weight(w) for w in matrices]
                else:
                    for mapped, weight in zip(previous, matrices):
                        epoch = mapped.program_epoch + 1
                        mapped.slice_data_imp(self.engine, weight.detach().float())
                        mapped.program_epoch = epoch
                self._weight_version = version
        return self._matrices

    def reset_mapping(self):
        """Call after changing a weight without a version update."""
        self._weight_version = None


class SimLinear(_MappedModule):
    def __init__(self, source, engine):
        super().__init__()
        self.engine = engine
        self.in_features, self.out_features = source.in_features, source.out_features
        self.weight, self.bias = source.weight, source.bias
        self.train(source.training)

    def forward(self, x):
        mapped = self._mapped([self.weight.T])[0]
        with torch.no_grad():
            result = self.engine.matmul(x.detach(), mapped).to(x.dtype)
        if torch.is_grad_enabled() and (x.requires_grad or self.weight.requires_grad):
            digital = F.linear(x, self.weight, None)
            result = digital + (result - digital).detach()
        return result if self.bias is None else result + self.bias


class SimConv2d(_MappedModule):
    def __init__(self, source, engine):
        super().__init__()
        if isinstance(source.padding, str):
            raise ValueError("Use explicit integer Conv2d padding")
        self.engine = engine
        self.weight, self.bias = source.weight, source.bias
        for name in ("in_channels", "out_channels", "groups", "kernel_size",
                     "stride", "padding", "dilation", "padding_mode"):
            setattr(self, name, getattr(source, name))
        self.train(source.training)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError("Conv2d expects NCHW input")
        original = x
        padding = self.padding
        if self.padding_mode != "zeros":
            x = F.pad(x, (padding[1], padding[1], padding[0], padding[0]), mode=self.padding_mode)
            padding = (0, 0)
        oh = (x.shape[2] + 2*padding[0] - self.dilation[0]*(self.kernel_size[0]-1)-1)//self.stride[0]+1
        ow = (x.shape[3] + 2*padding[1] - self.dilation[1]*(self.kernel_size[1]-1)-1)//self.stride[1]+1
        per_group = self.out_channels // self.groups
        weights = [w.flatten(1).T for w in self.weight.split(per_group)]
        mapped = self._mapped(weights)
        with torch.no_grad():
            patches = F.unfold(x.detach(), self.kernel_size, self.dilation, padding, self.stride)
            groups = patches.split(self.in_channels // self.groups * math.prod(self.kernel_size), dim=1)
            result = torch.cat([
                self.engine.matmul(part.transpose(1, 2), mat) for part, mat in zip(groups, mapped)
            ], dim=-1).transpose(1, 2).reshape(x.shape[0], self.out_channels, oh, ow).to(x.dtype)
        if torch.is_grad_enabled() and (original.requires_grad or self.weight.requires_grad):
            digital = F.conv2d(x, self.weight, None, self.stride, padding, self.dilation, self.groups)
            result = digital + (result-digital).detach()
        return result if self.bias is None else result + self.bias[None, :, None, None]


class SimGRU(nn.Module):
    """Standard PyTorch GRU equations with separately read input/recurrent arrays."""
    def __init__(self, source, engine):
        super().__init__()
        if source.bidirectional or source.dropout != 0:
            raise ValueError("GRU conversion supports unidirectional, zero-dropout models")
        self.input_size, self.hidden_size = source.input_size, source.hidden_size
        self.num_layers, self.batch_first = source.num_layers, source.batch_first
        self.input_linears, self.hidden_linears = nn.ModuleList(), nn.ModuleList()
        for layer in range(self.num_layers):
            for prefix, width, modules in (
                ("ih", self.input_size if layer == 0 else self.hidden_size, self.input_linears),
                ("hh", self.hidden_size, self.hidden_linears),
            ):
                weight = getattr(source, f"weight_{prefix}_l{layer}")
                linear = nn.Linear(width, 3*self.hidden_size, bias=source.bias,
                                   device=weight.device, dtype=weight.dtype)
                linear.weight = weight
                if source.bias:
                    linear.bias = getattr(source, f"bias_{prefix}_l{layer}")
                modules.append(SimLinear(linear, engine))
        self.train(source.training)

    def forward(self, x, hx=None):
        if x.ndim != 3:
            raise ValueError("GRU requires dense 3-D sequences, not PackedSequence")
        if not self.batch_first:
            x = x.transpose(0, 1)
        batch, steps, _ = x.shape
        if steps == 0:
            raise ValueError("Empty sequences are unsupported")
        if hx is not None and hx.shape != (self.num_layers, batch, self.hidden_size):
            raise ValueError("Invalid initial GRU state")
        hidden = ([x.new_zeros(batch, self.hidden_size) for _ in range(self.num_layers)]
                  if hx is None else list(hx.unbind()))
        finals = []
        for layer in range(self.num_layers):
            state, outputs = hidden[layer], []
            for t in range(steps):
                ir, iz, inn = self.input_linears[layer](x[:, t]).chunk(3, -1)
                hr, hz, hn = self.hidden_linears[layer](state).chunk(3, -1)
                reset, update = torch.sigmoid(ir+hr), torch.sigmoid(iz+hz)
                candidate = torch.tanh(inn + reset*hn)
                state = (1-update)*candidate + update*state
                outputs.append(state)
            x = torch.stack(outputs, dim=1)
            finals.append(state)
        return (x if self.batch_first else x.transpose(0, 1)), torch.stack(finals)


def convert_model(model, engine):
    """Convert Linear, Conv2d (including grouped/depthwise) and supported GRU.

    Call after moving the model to engine.device. Shared modules remain shared.
    Training uses a digital straight-through gradient and reprograms changed
    weights on their next call.
    """
    if not isinstance(engine, SimulationEngine):
        raise TypeError("Expected SimulationEngine")
    supported = (nn.Linear, nn.Conv2d, nn.GRU)
    items = [(name, module) for name, module in model.named_modules(remove_duplicate=False)
             if isinstance(module, supported)]
    for name, module in items:
        if isinstance(module, nn.Conv2d) and isinstance(module.padding, str):
            raise ValueError(f"{name}: string padding is unsupported; model unchanged")
        if isinstance(module, nn.GRU) and (module.bidirectional or module.dropout != 0):
            raise ValueError(f"{name}: bidirectional/dropout GRU unsupported; model unchanged")
        if next(module.parameters()).device != engine.device:
            raise ValueError(f"{name}: move model to {engine.device} before conversion")
    replaced = {}
    for name, module in items:
        if id(module) not in replaced:
            cls = SimLinear if isinstance(module, nn.Linear) else SimConv2d if isinstance(module, nn.Conv2d) else SimGRU
            replaced[id(module)] = cls(module, engine)
        replacement = replaced[id(module)]
        if not name:
            return replacement
        parent_name, _, attribute = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attribute, replacement)
    return model
