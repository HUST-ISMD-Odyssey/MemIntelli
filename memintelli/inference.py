"""Small inference helpers for the restricted v3.1 Triton profile."""
import torch

from .NN_layers import LinearMem
from .pimpy.triton_engine import TritonEngine


@torch.no_grad()
def linear_from_torch(layer, engine, *, streaming=False, free_weights=False):
    """Map a torch Linear without modifying the source module.

    Streaming reuses LinearMem's CPU/pinned-buffer lifecycle. This helper does
    not install a speculative prefetch order or promise a total-memory budget.
    """
    if not isinstance(layer, torch.nn.Linear) or not isinstance(engine, TritonEngine):
        raise TypeError("Expected torch.nn.Linear and TritonEngine")
    if layer.weight.dtype != torch.bfloat16:
        raise ValueError("The v3.1 Triton profile requires BF16 model weights")
    result = LinearMem(
        engine, layer.in_features, layer.out_features,
        [1] * engine.activation_bits, [1] * engine.weight_bits,
        bias=layer.bias is not None, device="cpu", dtype=torch.bfloat16,
        input_paral_size=(1, 64), weight_paral_size=(64, 64),
        input_quant_gran=(1, 64), weight_quant_gran=(64, 64),
        skip_initial_mapping=True,
    )
    result.weight.copy_(layer.weight.detach().cpu())
    if layer.bias is not None:
        result.bias.copy_(layer.bias.detach().cpu())
    result.requires_grad_(False)
    result._prepare_inference_weight(
        streaming=streaming, free_weights=free_weights, pin_policy="persistent"
    )
    return result.eval()


@torch.no_grad()
def replace_linears(model, engine, *, streaming=False, free_weights=True):
    """Replace every torch Linear, including lm_head; return replaced names.

    Accepts a model already placed on its execution device. Shared Linear
    objects remain shared after replacement. This conversion is inference-only.
    """
    if not isinstance(engine, TritonEngine):
        raise TypeError("Expected TritonEngine")
    originals = [(name, module) for name, module in model.named_modules(remove_duplicate=False)
                 if name and isinstance(module, torch.nn.Linear)]
    if not originals:
        raise ValueError("No child torch.nn.Linear modules found")
    for name, module in originals:
        if module.weight.dtype != torch.bfloat16:
            raise ValueError(f"{name}: weights must be BF16")
    replacements = {}
    for name, layer in originals:
        if id(layer) not in replacements:
            replacements[id(layer)] = linear_from_torch(
                layer, engine, streaming=streaming, free_weights=free_weights
            )
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, replacements[id(layer)])
    model.eval()
    return [name for name, _ in originals]
