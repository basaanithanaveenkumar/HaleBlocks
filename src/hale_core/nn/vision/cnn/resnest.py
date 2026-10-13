"""
ResNeSt Vision Backbones
=========================

Implements the ResNeSt family (Zhang et al. 2020, "ResNeSt: Split-Attention
Networks"). arXiv:2004.08955.

Architecture
------------
ResNeSt extends ResNet by replacing the 3×3 bottleneck convolution with a
**Split-Attention** convolution, which combines grouped convolutions with
attention across splits.

Key idea: split feature maps into `radix` branches, apply convolution to each,
then use a soft-max gating mechanism to recombine them:

  Input [B, C_in, H, W]
    ↓ Conv(C_in → out*radix, groups=groups)   # create radix feature splits
    ↓ reshape to [B, radix, C_out//groups, H, W] per group
    ↓ element-wise sum across radix → [B, C_out, H, W]
    ↓ GlobalAvgPool → [B, C_out, 1, 1]
    ↓ BN → Conv(C_out → C_out//4) → ReLU → Conv(C_out//4 → C_out*radix)
    ↓ reshape → softmax over radix axis → attention weights [B, radix, C_out//groups, 1, 1]
    ↓ weighted sum of splits → [B, C_out, H, W]

This is "attention across feature splits" — different from SE (channel attention
over pooled features) or CBAM (spatial + channel).  Split attention generalises
SKNet (Selective Kernel) to multiple groups.

ResNeSt Bottleneck: SplitAttentionConv2d replaces the 3×3 conv.
Stage structure (same as ResNet):
  Stem + 4 stages with depths depending on model size.

Feature map outputs:
  feature_maps[0] → [B, 256,  H/4,  W/4]
  feature_maps[1] → [B, 512,  H/8,  W/8]
  feature_maps[2] → [B, 1024, H/16, W/16]
  feature_maps[3] → [B, 2048, H/32, W/32]

Registered names
----------------
  resnest50    — depths=[3, 4, 6, 3],  ~27M params
  resnest101   — depths=[3, 4, 23, 3], ~48M params
  resnest200   — depths=[3, 24, 36, 3],~70M params
  resnest269   — depths=[3, 30, 48, 8],~110M params

Citation
--------
  Zhang, H. et al. (2020). ResNeSt: Split-Attention Networks. arXiv:2004.08955.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.registry import register_vision_backbone


# ── Split-Attention Conv2d ────────────────────────────────────────────────────

class SplitAttentionConv2d(nn.Module):
    """Convolution with split-attention (radix-wise gating).

    Produces `radix` feature map splits, softmax-weights them using a global
    attention branch, then returns the weighted sum.

    Args:
        in_channels:  Input channel count.
        out_channels: Output channel count (per split, before combining).
        kernel_size:  Conv kernel size.
        stride:       Spatial stride.
        padding:      Spatial padding.
        groups:       Number of convolution groups (cardinality).
        radix:        Number of splits (default 2).
        reduction:    Reduction factor for the attention FC bottleneck.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int = 3,
        stride:       int = 1,
        padding:      int = 1,
        groups:       int = 1,
        radix:        int = 2,
        reduction:    int = 4,
    ) -> None:
        super().__init__()
        inter_channels = max(in_channels * radix // reduction, 32)
        self.radix     = radix
        self.groups    = groups
        self.out_ch    = out_channels

        # Main conv: produces out_channels × radix feature maps
        self.conv = nn.Conv2d(
            in_channels, out_channels * radix,
            kernel_size=kernel_size, stride=stride, padding=padding,
            groups=groups * radix, bias=False,
        )
        self.bn0 = nn.BatchNorm2d(out_channels * radix)

        # Attention branch: globalpool → BN → FC → ReLU → FC → softmax
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn1  = nn.BatchNorm2d(out_channels)
        self.fc1  = nn.Conv2d(out_channels, inter_channels, kernel_size=1, groups=groups, bias=False)
        self.bn2  = nn.BatchNorm2d(inter_channels)
        self.fc2  = nn.Conv2d(inter_channels, out_channels * radix, kernel_size=1, groups=groups, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Produce all radix feature maps
        x   = F.relu(self.bn0(self.conv(x)), inplace=True)     # [B, C*radix, H, W]
        B, CR, H, W = x.shape
        C = self.out_ch

        if self.radix > 1:
            # Split into radix branches: [B, radix, C, H, W]
            splits = x.reshape(B, self.radix, C, H, W)
            # Sum over radix for the attention query
            gap = splits.sum(dim=1)                             # [B, C, H, W]
        else:
            gap = x

        # Attention branch
        gap = self.bn1(gap)
        gap = self.pool(gap)                                    # [B, C, 1, 1]
        gap = F.relu(self.bn2(self.fc1(gap)), inplace=True)    # [B, inter, 1, 1]
        atten = self.fc2(gap)                                   # [B, C*radix, 1, 1]

        if self.radix > 1:
            # Softmax over radix axis
            atten = atten.reshape(B, self.radix, C, 1, 1)
            atten = F.softmax(atten, dim=1)                     # [B, radix, C, 1, 1]
            # Weighted combination
            out = (splits * atten).sum(dim=1)                   # [B, C, H, W]
        else:
            out = x * torch.sigmoid(atten)

        return out.contiguous()


# ── ResNeSt Bottleneck ────────────────────────────────────────────────────────

class ResNeStBottleneck(nn.Module):
    """ResNeSt Bottleneck: 1×1 → SplitAttn3×3 → 1×1 with residual.

    Replaces the standard 3×3 conv of a ResNet Bottleneck with SplitAttentionConv2d.
    The 1×1 reduce and 1×1 expand convolutions are unchanged from ResNet.

    Args:
        in_planes:  Input channels.
        planes:     Bottleneck width (output = planes × 4).
        stride:     Stride for the split-attention conv.
        downsample: Projection shortcut (if needed).
        radix:      Number of split-attention branches.
        groups:     Convolution groups.
    """

    expansion: int = 4

    def __init__(
        self,
        in_planes:  int,
        planes:     int,
        stride:     int               = 1,
        downsample: Optional[nn.Module] = None,
        radix:      int               = 2,
        groups:     int               = 1,
    ) -> None:
        super().__init__()
        out_channels = planes * self.expansion

        # 1×1 reduce
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)

        # 3×3 split-attention (core innovation)
        self.conv2 = SplitAttentionConv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1,
            groups=groups, radix=radix,
        )

        # 1×1 expand
        self.conv3 = nn.Conv2d(planes, out_channels, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.conv2(out)                    # split-attention (already includes BN+ReLU)
        out = F.relu(out, inplace=True)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = F.relu(out + identity, inplace=True)
        return out


# ── Core ResNeSt model ────────────────────────────────────────────────────────

class ResNeSt(BaseVisionBackbone):
    """ResNeSt backbone with split-attention bottleneck blocks.

    Args:
        layers:     Blocks per stage (4-element list).
        radix:      Split-attention radix (default 2).
        groups:     Convolution groups (default 1).
        img_size, in_chans, drop_rate, out_indices: See BaseVisionBackbone.
    """

    arch_name:    str       = "resnest"
    embed_dim:    int       = 2048
    feature_dims: list[int] = [256, 512, 1024, 2048]

    def __init__(
        self,
        layers:      list[int],
        radix:       int  = 2,
        groups:      int  = 1,
        img_size:    int  = 224,
        in_chans:    int  = 3,
        drop_rate:   float = 0.0,
        out_indices: list[int] | None = None,
    ) -> None:
        super().__init__(img_size=img_size, in_chans=in_chans,
                         drop_rate=drop_rate, out_indices=out_indices)

        self._radix     = radix
        self._groups    = groups
        self._in_planes = 64

        # ── Stem ──────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ── 4 stages ──────────────────────────────────────────────────────────
        self.stage0 = self._make_stage(64,  layers[0], stride=1)
        self.stage1 = self._make_stage(128, layers[1], stride=2)
        self.stage2 = self._make_stage(256, layers[2], stride=2)
        self.stage3 = self._make_stage(512, layers[3], stride=2)

        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        self._stages = [self.stage0, self.stage1, self.stage2, self.stage3]
        self._init_weights()

    def _make_stage(
        self,
        planes:   int,
        n_blocks: int,
        stride:   int,
    ) -> nn.Sequential:
        out_channels = planes * ResNeStBottleneck.expansion
        downsample   = None

        if stride != 1 or self._in_planes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self._in_planes, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        blocks = [ResNeStBottleneck(
            self._in_planes, planes, stride, downsample,
            radix=self._radix, groups=self._groups,
        )]
        self._in_planes = out_channels

        for _ in range(1, n_blocks):
            blocks.append(ResNeStBottleneck(
                self._in_planes, planes,
                radix=self._radix, groups=self._groups,
            ))

        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.modules():
            if isinstance(m, ResNeStBottleneck):
                nn.init.zeros_(m.bn3.weight)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """Run ResNeSt forward pass.

        Args:
            x: [B, in_chans, H, W]

        Returns:
            BackboneOutput:
              pooled_output — [B, 2048]
              feature_maps  — [s0, s1, s2, s3] at strides [4, 8, 16, 32]
        """
        x = self.stem(x)

        s0 = self.stage0(x)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)

        pooled = self.dropout(self.pool(s3).flatten(1))

        all_maps = [s0, s1, s2, s3]
        if self.out_indices is not None:
            all_maps = [all_maps[i] for i in self.out_indices]

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=None,
            feature_maps=all_maps,
            attentions=None,
        )

    def freeze_up_to_stage(self, stage: int) -> None:
        """Freeze stem and stages 0..stage (inclusive)."""
        if stage < 0 or stage > 3:
            raise ValueError(f"stage must be in [0, 3], got {stage}")
        for p in self.stem.parameters():
            p.requires_grad_(False)
        for i in range(stage + 1):
            for p in self._stages[i].parameters():
                p.requires_grad_(False)


# ── Registered variants ───────────────────────────────────────────────────────

@register_vision_backbone("resnest50")
class ResNeSt50(ResNeSt):
    """ResNeSt-50: depths=[3,4,6,3], radix=2, ~27M params, embed_dim=2048."""
    arch_name    = "resnest50"
    embed_dim    = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__(layers=[3, 4, 6, 3], **kwargs)


@register_vision_backbone("resnest101")
class ResNeSt101(ResNeSt):
    """ResNeSt-101: depths=[3,4,23,3], radix=2, ~48M params, embed_dim=2048."""
    arch_name    = "resnest101"
    embed_dim    = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__(layers=[3, 4, 23, 3], **kwargs)


@register_vision_backbone("resnest200")
class ResNeSt200(ResNeSt):
    """ResNeSt-200: depths=[3,24,36,3], radix=2, ~70M params, embed_dim=2048."""
    arch_name    = "resnest200"
    embed_dim    = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__(layers=[3, 24, 36, 3], **kwargs)


@register_vision_backbone("resnest269")
class ResNeSt269(ResNeSt):
    """ResNeSt-269: depths=[3,30,48,8], radix=2, ~110M params, embed_dim=2048."""
    arch_name    = "resnest269"
    embed_dim    = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__(layers=[3, 30, 48, 8], **kwargs)
