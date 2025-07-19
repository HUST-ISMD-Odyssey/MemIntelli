# -*- coding:utf-8 -*-
# @File  : DeiT.py
# @Author: Zhou
# @Date  : 2024/4/1

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from typing import Optional, Union, Dict, Any
from memintelli.NN_layers import LinearMem

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
        "bw_e": bw_e
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
