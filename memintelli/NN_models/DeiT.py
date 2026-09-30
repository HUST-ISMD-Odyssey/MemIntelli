# -*- coding:utf-8 -*-
# @File  : DeiT.py
# @Author: Zhou
# @Date  : 2024/4/1

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Optional, Union, Dict, Any
from memintelli.NN_layers import LinearMem, tensor_bytes

# Pretrained model URLs
model_urls = {
    'deit_tiny_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
    'deit_small_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
    'deit_base_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
}

class DeiT(nn.Module):
    """
    Unified DeiT model with optional memristive mode

    Args:
        mem_enabled: Enable memristive mode
        mem_args: Dictionary containing memristive parameters
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        representation_size: int = None,
        distilled: bool = False,
        drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        mem_enabled: bool = False,
        mem_args: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.mem_enabled = mem_enabled
        self.mem_args = mem_args if self.mem_enabled else {}
        linear_layer = LinearMem if mem_enabled else nn.Linear

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                mem_enabled=mem_enabled,
                mem_args=self.mem_args
            ) for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                linear_layer(in_features=embed_dim, out_features=representation_size, **self.mem_args),
                nn.Tanh()
            )
        else:
            self.pre_logits = nn.Identity()

        self.head = linear_layer(in_features=self.num_features, out_features=num_classes, **self.mem_args) if num_classes > 0 else nn.Identity()
        if distilled:
            self.head_dist = linear_layer(in_features=embed_dim, out_features=num_classes, **self.mem_args)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1)
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.blocks(x)
        x = self.norm(x)
        x = self.pre_logits(x[:, 0])
        x = self.head(x)
        return x

    def update_weight(self):
        """Update memristive weights if enabled"""
        if not self.mem_enabled:
            return
        for m in self.modules():
            if isinstance(m, (LinearMem)):
                m.update_weight()


    def prepare_for_inference(self) -> None:
        """Prepare the model for optimized inference.

        Call after update_weight() and before inference.
        For lower peak memory, use update_weight_and_prepare() instead.
        """
        self.eval()
        if not self.mem_enabled:
            return
        import gc
        for module in self.modules():
            if isinstance(module, LinearMem):
                module.inference_mode = True
                module.weight_sliced.inference = True
                engine = module.engine
                if engine.write_variation == 0:
                    module.weight_sliced.compress_G(engine)
                module.weight_sliced.quantized_data = None
                module.weight_sliced.sliced_data = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def update_weight_and_prepare(self, streaming=False, free_weights: bool = True,
                                   gpu_memory_reserve: float = 4.0) -> None:
        """Combined update + prepare that minimizes peak GPU memory.

        Architecture: 3-phase pipeline (same as Qwen3).

        Phase 1: For each layer sequentially:
          weight → GPU → compute G → compress → offload G to pinned CPU → free weight

        Phase 2: Decide strategy and selectively load G back to GPU:
          - False:  load ALL G back to GPU (fastest inference)
          - True:   keep ALL on CPU, stream per-layer (lowest memory)
          - "auto": load as many as GPU allows, stream the rest (best tradeoff)

        Phase 3: Build async prefetch chain for streaming layers.

        Args:
            streaming: False | True | "auto" (see above)
            free_weights: If True (default), free nn.Parameter weight data after G.
            gpu_memory_reserve: GB reserved for activations when streaming="auto".
        """
        self.eval()
        if not self.mem_enabled:
            return
        import gc

        # ─── Phase 1: Compute G for all layers, ALWAYS offload to CPU immediately ───
        all_mem_layers = []
        layer_count = 0
        engine_device = None
        for module in self.modules():
            if isinstance(module, LinearMem):
                layer_count += 1
                engine = module.engine
                engine_device = engine.device

                module.weight_sliced.inference = True
                module.inference_mode = True
                module.update_weight()

                if engine.write_variation == 0:
                    module.weight_sliced.compress_G(engine)

                if free_weights:
                    module.weight.data = torch.empty(0, device='cpu', dtype=module.weight.dtype)

                module.weight_sliced.quantized_data = None
                module.weight_sliced.sliced_data = None

                if module.bias is not None and module.bias.device != engine_device:
                    module.bias.data = module.bias.data.to(engine_device)
                if module.input_slice_method.device != engine_device:
                    module.input_slice_method = module.input_slice_method.to(engine_device)
                if module.weight_slice_method.device != engine_device:
                    module.weight_slice_method = module.weight_slice_method.to(engine_device)

                # CRITICAL: Always offload G to pinned CPU immediately after computing!
                module._offload_to_cpu()

                all_mem_layers.append(module)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        if not all_mem_layers:
            return

        # ─── Phase 2: Decide streaming strategy and load GPU-resident layers ───
        layer_g_sizes = []
        total_g_bytes = 0
        for m in all_mem_layers:
            sz = 0
            for attr in ('G_indices', 'G', 'max_data', 'e_bias'):
                t = m._pinned_buffers.get(attr)
                sz += tensor_bytes(t)
            layer_g_sizes.append(sz)
            total_g_bytes += sz

        total_g_gb = total_g_bytes / 1024**3

        auto_mode = "auto_memory" if streaming == "auto" else streaming
        if auto_mode in ("auto_memory", "auto_speed") and torch.cuda.is_available():
            gpu_total = torch.cuda.get_device_properties(engine_device).total_memory
            gpu_used = torch.cuda.memory_allocated(engine_device)
            pending_cpu_bytes = sum(
                p.numel() * p.element_size()
                for p in self.parameters()
                if p.device.type == 'cpu' and p.numel() > 0
            )
            gpu_budget = (gpu_total - gpu_used
                          - int(gpu_memory_reserve * 1024**3)
                          - pending_cpu_bytes)
            budget_gb = gpu_budget / 1024**3
            pending_gb = pending_cpu_bytes / 1024**3
            print(f"[DeiT] {auto_mode}: GPU {gpu_total/1024**3:.1f}GB total, "
                  f"{gpu_used/1024**3:.1f}GB used, "
                  f"{pending_gb:.1f}GB pending, "
                  f"{gpu_memory_reserve:.0f}GB reserved → "
                  f"budget {budget_gb:.1f}GB for G")

            if total_g_bytes <= gpu_budget:
                streaming = False
                print(f"[DeiT] {auto_mode}: ALL G ({total_g_gb:.1f}GB) fits → GPU-resident (fastest)")
            else:
                indexed = sorted(
                    range(len(layer_g_sizes)),
                    key=lambda i: layer_g_sizes[i],
                    reverse=(auto_mode == "auto_memory"),
                )
                need_to_offload = max(0, total_g_bytes - gpu_budget)
                offloaded_bytes = 0
                offload_set = set()
                for idx in indexed:
                    if offloaded_bytes >= need_to_offload:
                        break
                    offload_set.add(idx)
                    offloaded_bytes += layer_g_sizes[idx]

                if offload_set:
                    max_stream_size = max(layer_g_sizes[i] for i in offload_set)
                    resident_bytes = total_g_bytes - offloaded_bytes
                    while resident_bytes + max_stream_size > gpu_budget:
                        resident_indices = [i for i in range(len(all_mem_layers))
                                            if i not in offload_set]
                        if not resident_indices:
                            break
                        selected = (
                            max(resident_indices, key=lambda i: layer_g_sizes[i])
                            if auto_mode == "auto_memory"
                            else min(resident_indices, key=lambda i: layer_g_sizes[i])
                        )
                        offload_set.add(selected)
                        offloaded_bytes += layer_g_sizes[selected]
                        resident_bytes -= layer_g_sizes[selected]
                        max_stream_size = max(layer_g_sizes[i] for i in offload_set)

                for i, m in enumerate(all_mem_layers):
                    if i in offload_set:
                        m._streaming = True
                    else:
                        m._load_to_device(engine_device)
                        m._pinned_buffers.clear()

                resident_count = layer_count - len(offload_set)
                resident_gb = (total_g_bytes - offloaded_bytes) / 1024**3
                print(f"[DeiT] {auto_mode}: partial: {resident_count} GPU-resident ({resident_gb:.1f}GB), "
                      f"{len(offload_set)} streaming ({offloaded_bytes/1024**3:.1f}GB)")
                streaming = "partial_done"

        if streaming is False:
            for m in all_mem_layers:
                m._load_to_device(engine_device)
                m._pinned_buffers.clear()
            print(f"[DeiT] Processed {layer_count} layers → GPU-resident "
                  f"({total_g_gb:.1f}GB G on GPU, weights {'freed' if free_weights else 'kept'})")
        elif streaming is True:
            for m in all_mem_layers:
                m._streaming = True
            print(f"[DeiT] Processed {layer_count} layers → full streaming "
                  f"({total_g_gb:.1f}GB G on CPU, weights {'freed' if free_weights else 'kept'})")

        # ─── Phase 3: Build async prefetch chain for streaming layers ───
        streaming_layers = [m for m in all_mem_layers if m._streaming]
        if streaming_layers:
            for i in range(len(streaming_layers) - 1):
                object.__setattr__(streaming_layers[i], '_next_streaming_layer', streaming_layers[i + 1])
            object.__setattr__(streaming_layers[-1], '_next_streaming_layer', streaming_layers[0])
            print(f"[DeiT] Async prefetch chain: {len(streaming_layers)} streaming layers linked")
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.,
        qkv_bias: bool = False,
        drop: float = 0.,
        attn_drop: float = 0.,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mem_enabled: bool = False,
        mem_args: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop, mem_enabled=mem_enabled, mem_args=mem_args)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, mem_enabled=mem_enabled, mem_args=mem_args)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Mlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int = None,
        out_features: int = None,
        act_layer: nn.Module = nn.GELU,
        drop: float = 0.,
        mem_enabled: bool = False,
        mem_args: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.mem_enabled = mem_enabled
        self.mem_args = mem_args if self.mem_enabled else {}
        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        linear_layer = LinearMem if mem_enabled else nn.Linear
        self.fc1 = linear_layer(in_features=in_features, out_features=hidden_features, **mem_args)
        self.act = act_layer()
        self.fc2 = linear_layer(in_features=hidden_features, out_features=out_features, **mem_args)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    """Standard attention implementation"""
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        mem_enabled: bool = False,
        mem_args: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        self.mem_enabled = mem_enabled
        self.mem_args = mem_args if self.mem_enabled else {}
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        linear_layer = LinearMem if mem_enabled else nn.Linear
        self.qkv = linear_layer(in_features=dim, out_features=dim * 3, bias=qkv_bias, **mem_args)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = linear_layer(in_features=dim, out_features=dim, **mem_args)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

def deit_zoo(
    model_name: str = 'deit_base_patch16_224',
    num_classes: int = 1000,
    pretrained: bool = False,
    mem_enabled: bool = False,
    engine: Optional[Any] = None,
    input_slice: Optional[Union[torch.Tensor, list]] = [1, 1, 2, 4],
    weight_slice: Optional[Union[torch.Tensor, list]] = [1, 1, 2, 4],
    device: Optional[Any] = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    bw_e: Optional[Any] = None,
    input_paral_size: Optional[Union[torch.Tensor, list]] = (1, 32),
    weight_paral_size: Optional[Union[torch.Tensor, list]] = (32, 32),
    input_quant_gran: Optional[Union[torch.Tensor, list]] = (1, 64),
    weight_quant_gran: Optional[Union[torch.Tensor, list]] = (64, 64)
) -> DeiT:
    """
    DeiT model factory

    Args:
        model_name (str): Model architecture name
        num_classes (int): Number of output classes
        pretrained (bool): Load pretrained weights
        mem_enabled (bool): Enable memristive mode
        engine (Optional[Any]): Memory engine for Mem layers
        input_slice (Optional[torch.Tensor, list]): Input tensor slicing configuration
        weight_slice (Optional[torch.Tensor, list]): Weight tensor slicing configuration
        device (Optional[Any]): Computation device (CPU/GPU)
        bw_e (Optional[Any]): if bw_e is None, the memristive engine is INT mode, otherwise, the memristive engine is FP mode (bw_e is the bitwidth of the exponent)
    """
    mem_args = {
        "engine": engine,
        "input_slice": input_slice,
        "weight_slice": weight_slice,
        "device": device,
        "bw_e": bw_e,
        "input_paral_size": input_paral_size,
        "weight_paral_size": weight_paral_size,
        "input_quant_gran": input_quant_gran,
        "weight_quant_gran": weight_quant_gran,
        "skip_initial_mapping": True,
    }
    mem_args = mem_args if mem_enabled else {}
    configs = {
        'deit_tiny_patch16_224': {'embed_dim': 192, 'depth': 12, 'num_heads': 3},
        'deit_small_patch16_224': {'embed_dim': 384, 'depth': 12, 'num_heads': 6},
        'deit_base_patch16_224': {'embed_dim': 768, 'depth': 12, 'num_heads': 12}
    }

    cfg = configs[model_name]
    model = DeiT(
        patch_size=16,
        num_classes=num_classes,
        mem_enabled=mem_enabled,
        mem_args=mem_args,
        **cfg
    )

    if pretrained:
        checkpoint = load_state_dict_from_url(model_urls[model_name])
        model.load_state_dict(checkpoint['model'])

    return model
