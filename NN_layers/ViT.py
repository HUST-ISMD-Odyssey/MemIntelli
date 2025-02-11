# -*- coding:utf-8 -*-
# @File  : ResNet.py
# @Author: Zhou
# @Date  : 2024/4/1

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
# from .convolution import Conv2dMem
# from .linear import LinearMem
#from .dataformat_layer import SlicedDataLayer
from NN_layers import Conv2dMem, LinearMem, SliceMethod
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
from typing import Any, Dict, List, Optional, Union
from torch.hub import load_state_dict_from_url


model_urls = {
    'deit_tiny_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
    'deit_small_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
    'deit_base_patch16_224': 'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
}

cfgs: Dict[str, Dict[str, Any]] = {
    'deit_tiny_patch16_224': {'patch_size': 16, 'embed_dim': 192, 'depth': 12, 'num_heads': 3, 'mlp_ratio': 4},
    'deit_small_patch16_224': {'patch_size': 16, 'embed_dim': 384, 'depth': 12, 'num_heads': 6, 'mlp_ratio': 4},
    'deit_base_patch16_224': {'patch_size': 16, 'embed_dim': 768, 'depth': 12, 'num_heads': 12, 'mlp_ratio': 4},
}

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

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

class DeiT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = (img_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                nn.Tanh()
            )
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

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

def deit_zoo(pretrained=False, model_name='deit_base_patch16_224', num_classes=1000, **kwargs):
    cfg = cfgs[model_name]
    model = DeiT(patch_size=cfg['patch_size'], embed_dim=cfg['embed_dim'], depth=cfg['depth'],
                 num_heads=cfg['num_heads'], mlp_ratio=cfg['mlp_ratio'], num_classes=num_classes, **kwargs)
    if pretrained:
        checkpoint = load_state_dict_from_url(model_urls[model_name], progress=True, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    return model



class AttentionMem(nn.Module):
    def __init__(self, engine, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 input_slice=None, weight_slice=None, device=None, bw_e=None, input_en=False, dbfp_en=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = LinearMem(engine, dim, dim * 3, bias=qkv_bias,
                             input_sli_med=input_slice, weight_sli_med=weight_slice,
                             device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = LinearMem(engine, dim, dim,
                              input_sli_med=input_slice, weight_sli_med=weight_slice,
                              device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MlpMem(nn.Module):
    def __init__(self, engine, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 input_slice=None, weight_slice=None, device=None, bw_e=None, input_en=False, dbfp_en=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = LinearMem(engine, in_features, hidden_features,
                             input_sli_med=input_slice, weight_sli_med=weight_slice,
                             device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
        self.act = act_layer()
        self.fc2 = LinearMem(engine, hidden_features, out_features,
                             input_sli_med=input_slice, weight_sli_med=weight_slice,
                             device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class BlockMem(nn.Module):
    def __init__(self, engine, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 input_slice=None, weight_slice=None, device=None, bw_e=None, input_en=False, dbfp_en=False,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionMem(engine, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
                                 input_slice=input_slice, weight_slice=weight_slice,
                                 device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MlpMem(engine, in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                          input_slice=input_slice, weight_slice=weight_slice,
                          device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class PatchEmbedMem(nn.Module):
    def __init__(self, engine, img_size=224, patch_size=16, in_chans=3, embed_dim=768,
                 input_slice=None, weight_slice=None, device=None, bw_e=None, input_en=False, dbfp_en=False):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = Conv2dMem(engine, in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size,
                              input_sli_med=input_slice, weight_sli_med=weight_slice,
                              device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class DeiTMem(nn.Module):
    def __init__(self, engine, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 input_slice=None, weight_slice=None, device=None, bw_e=None, input_en=False, dbfp_en=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        #self.patch_embed = PatchEmbedMem(engine, img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
                                        #  input_slice=input_slice, weight_slice=weight_slice,
                                        #  device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.blocks = nn.Sequential(*[
            BlockMem(engine, dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                     attn_drop=attn_drop_rate, input_slice=input_slice, weight_slice=weight_slice,
                     device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en)
            for _ in range(depth)])

        self.norm = nn.LayerNorm(embed_dim)

        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                LinearMem(engine, embed_dim, representation_size,
                          input_sli_med=input_slice, weight_sli_med=weight_slice,
                          device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en),
                nn.Tanh()
            )
        else:
            self.pre_logits = nn.Identity()

        self.head = LinearMem(engine, self.num_features, num_classes,
                              input_sli_med=input_slice, weight_sli_med=weight_slice,
                              device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = LinearMem(engine, self.embed_dim, self.num_classes,
                                       input_sli_med=input_slice, weight_sli_med=weight_slice,
                                       device=device, bw_e=bw_e, input_en=input_en, dbfp_en=dbfp_en) if num_classes > 0 else nn.Identity()

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
        for m in self.modules():
            if isinstance(m, LinearMem) or isinstance(m, Conv2dMem):
                m.update_weight()

def deit_zoo_mem(engine, input_slice, weight_slice, device, bw_e=None, input_en=False, dbfp_en=False, pretrained=False, model_name='deit_tiny_patch16_224', **kwargs):
    cfgs = {
        'deit_tiny_patch16_224': {'patch_size': 16, 'embed_dim': 192, 'depth': 12, 'num_heads': 3, 'mlp_ratio': 4},
        'deit_small_patch16_224': {'patch_size': 16, 'embed_dim': 384, 'depth': 12, 'num_heads': 6, 'mlp_ratio': 4},
        'deit_base_patch16_224': {'patch_size': 16, 'embed_dim': 768, 'depth': 12, 'num_heads': 12, 'mlp_ratio': 4},
    }
    
    cfg = cfgs[model_name]
    model = DeiTMem(
        engine=engine,
        patch_size=cfg['patch_size'],
        embed_dim=cfg['embed_dim'],
        depth=cfg['depth'],
        num_heads=cfg['num_heads'],
        mlp_ratio=cfg['mlp_ratio'],
        input_slice=input_slice,
        weight_slice=weight_slice,
        device=device,
        bw_e=bw_e,
        input_en=input_en,
        dbfp_en=dbfp_en,
        **kwargs
    )
    
    if pretrained:
        checkpoint = load_state_dict_from_url(model_urls[model_name], progress=True, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    
    return model