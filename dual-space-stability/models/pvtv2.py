"""
PVTv2-B2 backbone.

Adapted from the official PVT implementation
(https://github.com/whai362/PVT, Apache License 2.0), reduced to the encoder
used in this work. Only the B2 configuration is included, since that is the
backbone reported in the paper.

Pretrained weights
------------------
  https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth
  place the file at ./pretrained/pvt_v2_b2.pth

Feature map sizes for a 352x352 input:
  Stage 1: patch_embed(4) -> 3 blocks -> (B,  64, 88, 88)
  Stage 2: patch_embed(2) -> 4 blocks -> (B, 128, 44, 44)
  Stage 3: patch_embed(2) -> 6 blocks -> (B, 320, 22, 22)
  Stage 4: patch_embed(2) -> 3 blocks -> (B, 512, 11, 11)

Encoder parameters: about 25M.
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math


class DWConv(nn.Module):
    """Depth-wise convolution for positional encoding"""
    def __init__(self, dim=768):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=True)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    """Spatial Reduction Attention (SRA)"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0.,
                 proj_drop=0., sr_ratio=1, linear=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.linear = linear

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer Block with SRA"""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop,
                              sr_ratio=sr_ratio, linear=linear)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        # Drop path (stochastic depth)
        self.drop_path = nn.Identity()  # simplified, no stochastic depth

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class OverlapPatchEmbed(nn.Module):
    """Overlapping Patch Embedding"""
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PyramidVisionTransformerV2(nn.Module):
    """PVTv2 backbone returning four feature scales with channels = embed_dims."""
    def __init__(self, patch_size=7, in_chans=3, embed_dims=[64, 128, 320, 512],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
                 qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 norm_layer=partial(nn.LayerNorm, eps=1e-6),
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
                 linear=False):
        super().__init__()
        self.depths = depths
        self.embed_dims = embed_dims

        # Patch embeddings
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4,
                                              in_chans=in_chans, embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2,
                                              in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2,
                                              in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2,
                                              in_chans=embed_dims[2], embed_dim=embed_dims[3])

        # Transformer blocks
        self.block1 = nn.ModuleList([
            Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  norm_layer=norm_layer, sr_ratio=sr_ratios[0], linear=linear)
            for _ in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        self.block2 = nn.ModuleList([
            Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  norm_layer=norm_layer, sr_ratio=sr_ratios[1], linear=linear)
            for _ in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        self.block3 = nn.ModuleList([
            Block(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  norm_layer=norm_layer, sr_ratio=sr_ratios[2], linear=linear)
            for _ in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        self.block4 = nn.ModuleList([
            Block(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                  norm_layer=norm_layer, sr_ratio=sr_ratios[3], linear=linear)
            for _ in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

    def forward(self, x):
        outs = []

        # Stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1:
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(-1, H, W, self.embed_dims[0]).permute(0, 3, 1, 2)
        outs.append(x)

        # Stage 2
        x, H, W = self.patch_embed2(x)
        for blk in self.block2:
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(-1, H, W, self.embed_dims[1]).permute(0, 3, 1, 2)
        outs.append(x)

        # Stage 3
        x, H, W = self.patch_embed3(x)
        for blk in self.block3:
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(-1, H, W, self.embed_dims[2]).permute(0, 3, 1, 2)
        outs.append(x)

        # Stage 4
        x, H, W = self.patch_embed4(x)
        for blk in self.block4:
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(-1, H, W, self.embed_dims[3]).permute(0, 3, 1, 2)
        outs.append(x)

        return outs  # [stage1, stage2, stage3, stage4]


def pvt_v2_b2(**kwargs):
    """PVTv2-B2 configuration."""
    model = PyramidVisionTransformerV2(
        patch_size=7, embed_dims=[64, 128, 320, 512],
        num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
        **kwargs
    )
    return model


def load_pvtv2_pretrained(model, pretrained_path):
    """Load pretrained encoder weights, skipping keys that do not match."""
    if pretrained_path and os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location='cpu')
        # keep only keys whose name and shape match the encoder
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() 
                          if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"  Loaded {len(pretrained_dict)}/{len(model_dict)} params from {pretrained_path}")
    else:
        print(f"  [WARNING] Pretrained weights not found: {pretrained_path}")
    return model


if __name__ == '__main__':
    model = pvt_v2_b2()
    x = torch.randn(2, 3, 352, 352)
    outs = model(x)
    for i, o in enumerate(outs):
        print(f"Stage {i+1}: {o.shape}")
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"PVTv2-B2 params: {total:.2f}M")
