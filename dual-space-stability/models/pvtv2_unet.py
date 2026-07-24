"""
PVTv2-UNet: PVTv2-B2 encoder + U-Net decoder.

This is the default backbone used in the paper. The encoder is a Transformer
(LayerNorm rather than BatchNorm), so shallow BN-to-IN style normalisation does
not apply directly; an optional InstanceNorm can be inserted after the first two
stages instead (`use_style_in`).

`forward(x, return_features=True)` additionally returns the shallow features
[e1, e2] used by the optional feature-consistency component. The default
inference path is unchanged.

Pretrained encoder
------------------
  Download pvt_v2_b2.pth into ./pretrained/
  https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .pvtv2 import pvt_v2_b2, load_pvtv2_pretrained


class DecoderBlock(nn.Module):
    """U-Net decoder block."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear',
                                  align_corners=True)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class PVTv2UNet(nn.Module):
    """PVTv2-B2 encoder + U-Net decoder.

    Feature map sizes for a 352x352 input:
      Stage 1: (B, 64,  88,  88)   stride 4
      Stage 2: (B, 128, 44,  44)   stride 8
      Stage 3: (B, 320, 22,  22)   stride 16
      Stage 4: (B, 512, 11,  11)   stride 32

    The decoder upsamples from Stage 4, using skip connections with
    Stages 3, 2 and 1, then restores the input resolution.
    """

    def __init__(self, num_classes=1, pretrained_path=None, use_style_in=False):
        """
        Args:
            pretrained_path: path to pvt_v2_b2.pth
            use_style_in: insert InstanceNorm after the first two stages
        """
        super().__init__()
        self.use_style_in = use_style_in

        # Encoder: PVTv2-B2
        self.encoder = pvt_v2_b2()
        if pretrained_path:
            load_pvtv2_pretrained(self.encoder, pretrained_path)

        # optional shallow style normalisation; PVTv2 uses LayerNorm, so an
        # InstanceNorm is applied to the Stage 1 and Stage 2 feature maps instead
        if use_style_in:
            self.style_in1 = nn.InstanceNorm2d(64, affine=True)
            self.style_in2 = nn.InstanceNorm2d(128, affine=True)

        # Decoder
        # Stage4(512) → up + cat Stage3(320) → 256
        self.dec4 = DecoderBlock(512, 320, 256)
        # 256 → up + cat Stage2(128) → 128
        self.dec3 = DecoderBlock(256, 128, 128)
        # 128 → up + cat Stage1(64) → 64
        self.dec2 = DecoderBlock(128, 64, 64)
        # 64 -> up (no skip) -> 32
        self.dec1 = DecoderBlock(64, 0, 32)

        # output head
        self.final = nn.Conv2d(32, num_classes, 1)

    def forward(self, x, return_features=False):
        # encoder: four feature scales
        features = self.encoder(x)  # [s1, s2, s3, s4]
        e1, e2, e3, e4 = features

        # optional shallow style normalisation
        if self.use_style_in:
            e1 = self.style_in1(e1)
            e2 = self.style_in2(e2)

        # Decoder
        d4 = self.dec4(e4, e3)   # (B, 256, 22, 22)
        d3 = self.dec3(d4, e2)   # (B, 128, 44, 44)
        d2 = self.dec2(d3, e1)   # (B, 64,  88, 88)
        d1 = self.dec1(d2)       # (B, 32, 176, 176)

        # restore the input resolution
        out = self.final(d1)     # (B, 1, 176, 176)
        out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=True)

        if return_features:
            # shallow features, where appearance information is most present
            return out, [e1, e2]
        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # structural check without pretrained weights
    model = PVTv2UNet(num_classes=1, pretrained_path=None, use_style_in=True).to(device)

    x = torch.randn(2, 3, 352, 352).to(device)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")

    out2, feats = model(x, return_features=True)
    print(f"return_features: out={tuple(out2.shape)}, "
          f"feats={[tuple(f.shape) for f in feats]}")

    total = sum(p.numel() for p in model.parameters()) / 1e6
    enc = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    print(f"Params: {total:.2f}M total | Encoder: {enc:.2f}M | Decoder: {total-enc:.2f}M")
