"""
RegNet Vision Backbones
=======================

Implements the RegNet family (Radosavovic et al. 2020, "Designing Network Design
Spaces", CVPR 2020, https://arxiv.org/abs/2003.13678).

Key idea
--------
RegNet is not a single architecture but a principled *design space* for CNNs.
The authors analyse a large random population of AnyNet models and discover that
the best architectures share a simple structural regularity:

  - Widths grow approximately as a geometric sequence: w_j ≈ w_0 * α^j.
  - Depths, once quantised, produce a "RegNet" model (quantised design space).
  - A single group width g is shared across all stages.
  - Bottleneck ratio b=1 (no bottleneck; input and output widths match).

Two sub-families:
  RegNetX — no Squeeze-Excite (plain grouped bottleneck).
  RegNetY — with Squeeze-Excite (channel attention), typically ~0.5–1% better.

Block structure (ResBottleneckBlock)
-------------------------------------
  Conv 1×1 + BN + ReLU
  GroupConv 3×3 (groups = w / g) + BN + ReLU
  [SE if RegNetY]
  Conv 1×1 + BN
  + residual (projected shortcut if shapes differ)
  ReLU

Stem
----
  Conv(3→32, 3×3, stride=2) + BN + ReLU  → H/2 × W/2

Stages: stride-2 downsampling at the first block of each stage (from stage 1).

Registered names (GFlops × 100 suffix)
-----------------------------------------
  regnetx_002 … regnetx_320   (10 variants, no SE)
  regnety_002 … regnety_320   (10 variants, with SE)

Citation
--------
  Radosavovic, I., Kosaraju, R. P., Girshick, R., He, K., & Dollar, P. (2020).
  Designing Network Design Spaces. CVPR.
  arXiv:2003.13678
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.registry import register_vision_backbone


# ── Squeeze-Excite ────────────────────────────────────────────────────────────

class SqueezeExcite(nn.Module):
    """Channel attention: avgpool → FC(squeeze) → ReLU → FC(excite) → sigmoid."""

    def __init__(self, in_channels: int, reduction_ratio: int = 4) -> None:
        super().__init__()
        mid = max(1, in_channels // reduction_ratio)
        self.fc1 = nn.Linear(in_channels, mid)
        self.fc2 = nn.Linear(mid, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=[2, 3])                # [B, C]
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale.unsqueeze(-1).unsqueeze(-1)


# ── ResBottleneckBlock ────────────────────────────────────────────────────────

class ResBottleneckBlock(nn.Module):
    """Standard RegNet block (bottleneck_ratio=1 so in==out width).

    Args:
        in_channels:   Input channels.
        out_channels:  Output channels.
        stride:        Spatial stride (1 or 2).
        group_width:   Group width for the 3×3 grouped conv.
        use_se:        Whether to add Squeeze-Excite attention.
        se_reduction:  SE squeeze reduction factor.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        stride:       int  = 1,
        group_width:  int  = 1,
        use_se:       bool = False,
        se_reduction: int  = 4,
    ) -> None:
        super().__init__()
        groups = max(1, out_channels // group_width)

        # 1×1 proj
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)

        # 3×3 grouped conv
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3,
                               stride=stride, padding=1, groups=groups, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Channel attention (RegNetY only)
        self.se = SqueezeExcite(out_channels, se_reduction) if use_se else nn.Identity()

        # 1×1 expansion
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels)

        # Shortcut projection when shape changes
        self.shortcut: Optional[nn.Module] = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x if self.shortcut is None else self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)),  inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.se(out)
        out = self.bn3(self.conv3(out))

        return F.relu(out + skip, inplace=True)


# ── RegNet backbone ───────────────────────────────────────────────────────────

class RegNet(BaseVisionBackbone):
    """RegNet: principled network from quantised design space.

    Args:
        depths:      Number of blocks in each of the 4 stages.
        widths:      Channel width at each of the 4 stages (must be divisible by group_width).
        group_width: Shared group width across all stages.
        use_se:      If True, add Squeeze-Excite (RegNetY); else RegNetX.
        img_size:    Input resolution.
        in_chans:    Input channels.
        drop_rate:   Dropout after global pool.
        out_indices: Which stage feature maps to return.
    """

    def __init__(
        self,
        depths:      list[int],
        widths:      list[int],
        group_width: int,
        use_se:      bool       = False,
        img_size:    int        = 224,
        in_chans:    int        = 3,
        drop_rate:   float      = 0.0,
        out_indices: list[int] | None = None,
    ) -> None:
        super().__init__(img_size=img_size, in_chans=in_chans, drop_rate=drop_rate, out_indices=out_indices)

        self.embed_dim    = widths[-1]
        self.feature_dims = list(widths)

        # Stem: Conv(3→32, 3×3, s2) + BN + ReLU
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 4 stages with progressive downsampling
        self.stages = nn.ModuleList()
        in_ch = 32
        for stage_idx, (depth, width) in enumerate(zip(depths, widths)):
            blocks = []
            for i in range(depth):
                stride = 2 if (i == 0 and stage_idx > 0) else 1
                blocks.append(ResBottleneckBlock(
                    in_channels=in_ch,
                    out_channels=width,
                    stride=stride,
                    group_width=group_width,
                    use_se=use_se,
                ))
                in_ch = width
            self.stages.append(nn.Sequential(*blocks))

        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """RegNet forward: stem → 4 stages → global pool.

        Returns:
            BackboneOutput with pooled_output [B, widths[-1]] and 4 feature maps.
        """
        x = self.stem(x)  # [B, 32, H/2, W/2]

        feature_maps = []
        for stage in self.stages:
            x = stage(x)
            feature_maps.append(x)

        pooled = self.dropout(self.pool(feature_maps[-1]).flatten(1))

        if self.out_indices is not None:
            feature_maps = [feature_maps[i] for i in self.out_indices]

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=None,
            feature_maps=feature_maps,
            attentions=None,
        )

    def freeze_up_to_stage(self, stage: int) -> None:
        """Freeze stem and stages 0 through `stage` inclusive."""
        if stage < 0 or stage > 3:
            raise ValueError(f"stage must be in [0, 3], got {stage}")
        for p in self.stem.parameters():
            p.requires_grad_(False)
        for i in range(stage + 1):
            for p in self.stages[i].parameters():
                p.requires_grad_(False)


# ── Stage config tables ───────────────────────────────────────────────────────

# From RegNet paper Table 14 (RegNetX) — (depths, widths, group_width)
_REGNETX_CFGS: dict[str, dict] = {
    "002": dict(depths=[1, 1, 4,  7], widths=[24,  56,  152,  368], group_width=8),
    "004": dict(depths=[1, 2, 7, 12], widths=[32,  64,  160,  384], group_width=16),
    "008": dict(depths=[1, 3, 8,  2], widths=[64, 128,  288,  672], group_width=16),
    "016": dict(depths=[2, 4, 10, 2], widths=[72, 168,  408,  912], group_width=24),
    "032": dict(depths=[2, 6, 15, 2], widths=[96, 192,  432, 1008], group_width=48),
    "064": dict(depths=[2, 4, 10, 1], widths=[168, 392,  784, 1624], group_width=56),
    "080": dict(depths=[2, 5, 14, 2], widths=[80, 240,  720, 1920], group_width=120),
    "120": dict(depths=[2, 5, 11, 1], widths=[224, 448,  896, 2240], group_width=112),
    "160": dict(depths=[2, 6, 13, 1], widths=[256, 512,  896, 2048], group_width=128),
    "320": dict(depths=[2, 7, 13, 1], widths=[336, 672, 1344, 2520], group_width=168),
}


def _make_regnetx(key: str, name: str) -> type:
    cfg = _REGNETX_CFGS[key]
    w   = cfg["widths"][-1]

    class _RegNetX(RegNet):
        arch_name    = name
        embed_dim    = w
        feature_dims = cfg["widths"]

        def __init__(self, **kwargs):
            super().__init__(
                depths=cfg["depths"],
                widths=cfg["widths"],
                group_width=cfg["group_width"],
                use_se=False,
                **kwargs,
            )

    _RegNetX.__name__ = _RegNetX.__qualname__ = name.replace("_", "").upper()
    return _RegNetX


def _make_regnety(key: str, name: str) -> type:
    cfg = _REGNETX_CFGS[key]  # same structural config; only differs in SE
    w   = cfg["widths"][-1]

    class _RegNetY(RegNet):
        arch_name    = name
        embed_dim    = w
        feature_dims = cfg["widths"]

        def __init__(self, **kwargs):
            super().__init__(
                depths=cfg["depths"],
                widths=cfg["widths"],
                group_width=cfg["group_width"],
                use_se=True,
                **kwargs,
            )

    _RegNetY.__name__ = _RegNetY.__qualname__ = name.replace("_", "").upper()
    return _RegNetY


# Programmatically register all 20 variants
_SUFFIXES = ["002", "004", "008", "016", "032", "064", "080", "120", "160", "320"]

for _sfx in _SUFFIXES:
    _rx_name = f"regnetx_{_sfx}"
    _ry_name = f"regnety_{_sfx}"
    _rx_cls  = _make_regnetx(_sfx, _rx_name)
    _ry_cls  = _make_regnety(_sfx, _ry_name)
    register_vision_backbone(_rx_name)(_rx_cls)
    register_vision_backbone(_ry_name)(_ry_cls)
    # Also expose in module namespace so imports work
    globals()[_rx_cls.__name__] = _rx_cls
    globals()[_ry_cls.__name__] = _ry_cls
