"""
ResNeXt Vision Backbones
========================

Implements the ResNeXt family (Xie et al. CVPR 2017, "Aggregated Residual
Transformations for Deep Neural Networks").

The key insight: instead of a single "wider" convolution, use a group of
"narrower" convolutions (cardinality C) — aggregated transformations that
are computationally equivalent to a grouped convolution.  This improves
accuracy without increasing FLOPs, adding a new dimension to scaling beyond
depth and width.

Architecture
------------
ResNeXt uses the same 4-stage hierarchy as ResNet but replaces the 3×3 conv
in each Bottleneck with a grouped convolution:

  ResNeXtBottleneck (32×4d, i.e. 32 groups × 4 width):
    1×1 Conv(in → C*D)          — reduce to grouped width
    3×3 GroupConv(C*D → C*D, groups=C) — aggregated transformations
    1×1 Conv(C*D → planes×4)   — expand to output channels
    + residual shortcut

  where C = cardinality (groups), D = base_width per group
  For 32×4d: C=32, D=4, so mid_ch = C*D = 128
  For 32×8d: C=32, D=8, so mid_ch = C*D = 256

Stage structure identical to ResNet:
  Stem → Stage 0 (64→256) → Stage 1 (128→512) → Stage 2 (256→1024) → Stage 3 (512→2048)
  embed_dim = 2048 for all variants (same as ResNet50/101).

Registered names
----------------
  resnext50_32x4d   — [3, 4, 6, 3]  × ResNeXtBottleneck(C=32, D=4)
  resnext101_32x4d  — [3, 4, 23, 3] × ResNeXtBottleneck(C=32, D=4)
  resnext101_32x8d  — [3, 4, 23, 3] × ResNeXtBottleneck(C=32, D=8)

Citation
--------
  Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2017). Aggregated
  Residual Transformations for Deep Neural Networks. CVPR 2017. arXiv:1611.05431.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.registry import register_vision_backbone


# ── ResNeXt Bottleneck ────────────────────────────────────────────────────────

class ResNeXtBottleneck(nn.Module):
    """ResNeXt Bottleneck with grouped (cardinality) convolution.

    The key difference from standard Bottleneck: the 3×3 conv uses groups=C
    (cardinality), which implements C parallel transformations of equal width.
    Mathematically equivalent to a width-C grouped conv but much more expressive
    than a single wider conv at the same parameter budget.

    Args:
        in_planes:   Input channels.
        planes:      Stage "planes" (output = planes * expansion = planes * 4).
        stride:      Stride for the 3×3 grouped conv.
        cardinality: Number of transformation groups (default 32).
        base_width:  Width per group in the bottleneck (default 4).
        downsample:  Optional projection shortcut.
    """
    expansion: int = 4

    def __init__(
        self,
        in_planes:   int,
        planes:      int,
        stride:      int = 1,
        cardinality: int = 32,
        base_width:  int = 4,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        # Bottleneck width = C * D; at the standard stage channel counts this
        # gives much wider intermediate features than ResNet bottleneck at similar cost
        mid_ch = int(planes * (base_width / 64.0)) * cardinality

        self.conv1 = nn.Conv2d(in_planes, mid_ch, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid_ch)

        # Grouped conv: groups=cardinality implements the aggregated transformation
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, kernel_size=3, stride=stride,
                               padding=1, groups=cardinality, bias=False)
        self.bn2   = nn.BatchNorm2d(mid_ch)

        out_ch = planes * self.expansion
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_ch)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)


# ── Core ResNeXt model ────────────────────────────────────────────────────────

class ResNeXt(BaseVisionBackbone):
    """Generic ResNeXt backbone.

    Same 4-stage structure as ResNet but using grouped bottleneck blocks.

    Args:
        layers:      Number of blocks per stage [s0, s1, s2, s3].
        cardinality: Transformation groups (C in the paper).
        base_width:  Width per group at stage channels (D in the paper).
        img_size:    Input resolution.
        in_chans:    Input channels.
        drop_rate:   Dropout after global pooling.
    """

    embed_dim    = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(
        self,
        layers:      list[int],
        cardinality: int   = 32,
        base_width:  int   = 4,
        img_size:    int   = 224,
        in_chans:    int   = 3,
        drop_rate:   float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(img_size=img_size, in_chans=in_chans, drop_rate=drop_rate)

        self._cardinality = cardinality
        self._base_width  = base_width
        self._in_planes   = 64

        # Stem: same as ResNet — 7×7 conv → BN → ReLU → MaxPool
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.stage0 = self._make_stage( 64, layers[0], stride=1)
        self.stage1 = self._make_stage(128, layers[1], stride=2)
        self.stage2 = self._make_stage(256, layers[2], stride=2)
        self.stage3 = self._make_stage(512, layers[3], stride=2)

        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        self._stages = [self.stage0, self.stage1, self.stage2, self.stage3]
        self._init_weights()

    def _make_stage(self, planes: int, n_blocks: int, stride: int) -> nn.Sequential:
        out_ch = planes * ResNeXtBottleneck.expansion
        downsample = None
        if stride != 1 or self._in_planes != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(self._in_planes, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        blocks = [ResNeXtBottleneck(
            self._in_planes, planes, stride,
            cardinality=self._cardinality,
            base_width=self._base_width,
            downsample=downsample,
        )]
        self._in_planes = out_ch

        for _ in range(1, n_blocks):
            blocks.append(ResNeXtBottleneck(
                self._in_planes, planes,
                cardinality=self._cardinality,
                base_width=self._base_width,
            ))
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Zero-init last BN in each bottleneck for training stability
        for m in self.modules():
            if isinstance(m, ResNeXtBottleneck):
                nn.init.zeros_(m.bn3.weight)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        x  = self.stem(x)

        s0 = self.stage0(x)   # [B, 256,  H/4,  W/4]
        s1 = self.stage1(s0)  # [B, 512,  H/8,  W/8]
        s2 = self.stage2(s1)  # [B, 1024, H/16, W/16]
        s3 = self.stage3(s2)  # [B, 2048, H/32, W/32]

        pooled = self.dropout(self.pool(s3).flatten(1))  # [B, 2048]

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
        if stage < 0 or stage > 3:
            raise ValueError(f"stage must be in [0, 3], got {stage}")
        for p in self.stem.parameters():
            p.requires_grad_(False)
        for i in range(stage + 1):
            for p in self._stages[i].parameters():
                p.requires_grad_(False)


# ── Registered variants ───────────────────────────────────────────────────────

@register_vision_backbone("resnext50_32x4d")
class ResNeXt50_32x4d(ResNeXt):
    """ResNeXt-50 (32×4d): Bottleneck×[3,4,6,3], C=32, D=4, embed_dim=2048."""
    arch_name    = "resnext50_32x4d"
    embed_dim    = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs):
        super().__init__([3, 4, 6, 3], cardinality=32, base_width=4, **kwargs)


@register_vision_backbone("resnext101_32x4d")
class ResNeXt101_32x4d(ResNeXt):
    """ResNeXt-101 (32×4d): Bottleneck×[3,4,23,3], C=32, D=4, embed_dim=2048."""
    arch_name    = "resnext101_32x4d"
    embed_dim    = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs):
        super().__init__([3, 4, 23, 3], cardinality=32, base_width=4, **kwargs)


@register_vision_backbone("resnext101_32x8d")
class ResNeXt101_32x8d(ResNeXt):
    """ResNeXt-101 (32×8d): Bottleneck×[3,4,23,3], C=32, D=8, embed_dim=2048."""
    arch_name    = "resnext101_32x8d"
    embed_dim    = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs):
        super().__init__([3, 4, 23, 3], cardinality=32, base_width=8, **kwargs)
