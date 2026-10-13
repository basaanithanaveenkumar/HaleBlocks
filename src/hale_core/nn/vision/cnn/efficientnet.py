"""
EfficientNetV2 Vision Backbones (Tan & Le 2021)
=================================================

Reference: "EfficientNetV2: Smaller Models and Faster Training"
           https://arxiv.org/abs/2104.00298

Architecture Overview
---------------------
EfficientNetV2 builds on EfficientNet by replacing the initial MBConv blocks
with Fused-MBConv blocks, which are faster on modern accelerators.

Two block types:
  MBConv (Mobile Inverted Bottleneck Conv):
    input → expand 1×1 (ratio r) → depthwise 3×3 or 5×5 → SE → project 1×1
    Used in later stages where depthwise is more efficient.

  FusedMBConv:
    input → fused_expand 3×3 (ratio r, no depthwise separation) → SE → project 1×1
    Used in early stages where tensor sizes are large (channel count small).
    Fusing expand+depthwise into a single 3×3 avoids the bandwidth bottleneck
    of large input feature maps.

Squeeze-Excitation (SE):
  Global avg pool → fc1 (C//se_ratio) → SiLU → fc2 (C) → Sigmoid → multiply input.
  Recalibrates channel-wise feature responses adaptively.

Stochastic Depth (DropPath):
  Randomly drops entire residual branches during training (survival probability
  linearly interpolated from 1.0 at layer 0 to (1 - drop_path_rate) at last layer).
  Acts as a regularizer and improves accuracy.

Stage structure (for S variant):
  Stage 0: FusedMBConv, r=1, k=3, stride=1, channels=24,  n=2
  Stage 1: FusedMBConv, r=4, k=3, stride=2, channels=48,  n=4
  Stage 2: FusedMBConv, r=4, k=3, stride=2, channels=64,  n=4
  Stage 3: MBConv,      r=4, k=3, stride=2, channels=128, n=6
  Stage 4: MBConv,      r=6, k=3, stride=1, channels=160, n=9
  Stage 5: MBConv,      r=6, k=3, stride=2, channels=256, n=15
  Head:    Conv1×1 → 1280-dim

Feature pyramid:
  feature_maps[0] = stride-4 output (after stem)
  feature_maps[1] = stride-8 output (after stage 1 down)
  feature_maps[2] = stride-16 output (after stage 2 down + stage 3)
  feature_maps[3] = stride-32 output (after stage 5)

Registered variants:
  efficientnet_v2_s   — small
  efficientnet_v2_m   — medium
  efficientnet_v2_l   — large
  efficientnet_v2_xl  — extra-large
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.registry import register_vision_backbone


# ── Stochastic Depth ──────────────────────────────────────────────────────────

class StochasticDepth(nn.Module):
    """Drop entire residual branches with probability `drop_prob`.

    At training time, each sample in a batch has probability `drop_prob` of
    having its residual branch zeroed out (the skip connection always remains).
    At test time, the branch is always active (scaled by survival_prob).

    This is the "per-sample" stochastic depth from Huang et al. 2016.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        survival = 1.0 - self.drop_prob
        # Shape [B, 1, 1, 1] for broadcasting over spatial dims
        noise = torch.empty(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype)
        noise = noise.bernoulli_(survival)
        if survival > 0.0:
            noise.div_(survival)   # scale to maintain expected value
        return x * noise


# ── Squeeze-Excitation ────────────────────────────────────────────────────────

class SqueezeExcite(nn.Module):
    """Channel attention via global pooling followed by two fully-connected layers.

    SE reduces channels by se_ratio, then re-expands to original count.
    Output is used to SCALE the input features (not add to them).

    Args:
        in_chs:   Input channel count.
        se_ratio: Reduction ratio for the bottleneck fc layer (default 0.25 → C//4).
    """

    def __init__(self, in_chs: int, se_ratio: float = 0.25) -> None:
        super().__init__()
        se_chs = max(1, int(in_chs * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1  = nn.Conv2d(in_chs, se_chs, kernel_size=1)
        self.fc2  = nn.Conv2d(se_chs, in_chs, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.pool(x)
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


# ── Block Types ───────────────────────────────────────────────────────────────

class FusedMBConv(nn.Module):
    """Fused Mobile Inverted Bottleneck Convolution.

    Replaces expand_1x1 + depthwise_3x3 with a single fused_3x3 convolution.
    Faster on modern accelerators for small-channel early stages.

    Args:
        in_chs:        Input channels.
        out_chs:       Output channels.
        stride:        Spatial stride.
        expand_ratio:  Channel expansion factor before projection.
        se_ratio:      Squeeze-Excite reduction ratio (0 = no SE).
        drop_path:     Stochastic depth probability.
    """

    def __init__(
        self,
        in_chs:       int,
        out_chs:      int,
        stride:       int   = 1,
        expand_ratio: int   = 1,
        se_ratio:     float = 0.0,
        drop_path:    float = 0.0,
    ) -> None:
        super().__init__()
        mid_chs = _make_divisible(in_chs * expand_ratio, 8)
        has_skip = stride == 1 and in_chs == out_chs

        layers: list[nn.Module] = []

        if expand_ratio != 1:
            # Expand conv (fused 3×3 instead of 1×1 + depthwise)
            layers += [
                nn.Conv2d(in_chs, mid_chs, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(mid_chs),
                nn.SiLU(inplace=True),
            ]
        else:
            mid_chs = in_chs

        if se_ratio > 0.0:
            layers.append(SqueezeExcite(mid_chs, se_ratio))

        # Project conv (1×1)
        layers += [
            nn.Conv2d(mid_chs, out_chs, 1, bias=False),
            nn.BatchNorm2d(out_chs),
        ]
        if expand_ratio != 1:
            # Only add SiLU after project when no expansion (r=1 case)
            pass

        self.conv = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(drop_path) if has_skip else nn.Identity()
        self.has_skip = has_skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_skip:
            return x + self.stochastic_depth(self.conv(x))
        return self.conv(x)


class MBConv(nn.Module):
    """Standard Mobile Inverted Bottleneck Conv for EfficientNetV2.

    Separate expand 1×1 + depthwise k×k + project 1×1 with SE.

    Args:
        in_chs:        Input channels.
        out_chs:       Output channels.
        stride:        Spatial stride.
        expand_ratio:  Channel expansion factor.
        kernel_size:   Depthwise kernel size (3 or 5).
        se_ratio:      Squeeze-Excite reduction ratio.
        drop_path:     Stochastic depth probability.
    """

    def __init__(
        self,
        in_chs:       int,
        out_chs:      int,
        stride:       int   = 1,
        expand_ratio: int   = 6,
        kernel_size:  int   = 3,
        se_ratio:     float = 0.25,
        drop_path:    float = 0.0,
    ) -> None:
        super().__init__()
        mid_chs  = _make_divisible(in_chs * expand_ratio, 8)
        has_skip = stride == 1 and in_chs == out_chs
        pad = (kernel_size - 1) // 2

        self.conv = nn.Sequential(
            # Expand
            nn.Conv2d(in_chs, mid_chs, 1, bias=False),
            nn.BatchNorm2d(mid_chs),
            nn.SiLU(inplace=True),
            # Depthwise
            nn.Conv2d(mid_chs, mid_chs, kernel_size, stride=stride,
                      padding=pad, groups=mid_chs, bias=False),
            nn.BatchNorm2d(mid_chs),
            nn.SiLU(inplace=True),
            # Squeeze-Excite
            SqueezeExcite(mid_chs, se_ratio),
            # Project
            nn.Conv2d(mid_chs, out_chs, 1, bias=False),
            nn.BatchNorm2d(out_chs),
        )
        self.stochastic_depth = StochasticDepth(drop_path) if has_skip else nn.Identity()
        self.has_skip = has_skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_skip:
            return x + self.stochastic_depth(self.conv(x))
        return self.conv(x)


# ── Stage Config ──────────────────────────────────────────────────────────────

from dataclasses import dataclass

@dataclass
class _StageConfig:
    """Configuration for one EfficientNetV2 stage."""
    block_type:   str    # "fused" or "mbconv"
    in_chs:       int
    out_chs:      int
    stride:       int
    expand_ratio: int
    kernel_size:  int
    num_blocks:   int
    se_ratio:     float  # 0 for fused, 0.25 for mbconv


# ── Main Backbone ─────────────────────────────────────────────────────────────

class EfficientNetV2Backbone(BaseVisionBackbone):
    """EfficientNetV2 backbone returning multi-scale feature maps.

    Multi-scale output:
      feature_maps[0]  stride-4   (after stem)
      feature_maps[1]  stride-8   (after stage 1)
      feature_maps[2]  stride-16  (after stages 2+3)
      feature_maps[3]  stride-32  (after stages 4+5)

    Args:
        stages:          List of _StageConfig (defines the full architecture).
        head_channels:   Output channels after the head 1×1 conv.
        drop_path_rate:  Max stochastic depth rate (linearly scaled per block).
        img_size, in_chans, drop_rate: See BaseVisionBackbone.
    """

    arch_name:    str       = "efficientnet_v2"
    embed_dim:    int       = 1280
    feature_dims: list[int] = []

    def __init__(
        self,
        stages:         list[_StageConfig],
        head_channels:  int  = 1280,
        drop_path_rate: float = 0.0,
        img_size:       int  = 224,
        in_chans:       int  = 3,
        drop_rate:      float = 0.0,
        out_indices:    list[int] | None = None,
    ) -> None:
        self.embed_dim = head_channels

        # Feature dims at strides 4, 8, 16, 32 — to be filled below
        super().__init__(img_size=img_size, in_chans=in_chans,
                         drop_rate=drop_rate, out_indices=out_indices)

        # Count total blocks for stochastic depth schedule
        total_blocks = sum(s.num_blocks for s in stages)
        block_idx    = 0

        # ── Stem (stride-2, produces H/2) ────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, stages[0].in_chs, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(stages[0].in_chs),
            nn.SiLU(inplace=True),
        )

        # ── Stages ───────────────────────────────────────────────────────────
        # We group stages by their STRIDE to build a 4-scale feature pyramid.
        # stride-2 transitions mark new pyramid levels.
        self.stage_groups: nn.ModuleList = nn.ModuleList()
        stride_4_chs  = None  # filled on first stride-2 stage
        stride_8_chs  = None
        stride_16_chs = None
        stride_32_chs = None
        current_stride = 2

        groups: list[list[nn.Module]] = []
        current_group: list[nn.Module] = []
        last_out_chs = stages[0].in_chs

        for s in stages:
            # dp_rate linearly increases from 0 to drop_path_rate
            for i in range(s.num_blocks):
                dp = drop_path_rate * block_idx / max(total_blocks - 1, 1)
                stride_i = s.stride if i == 0 else 1   # stride only on first block

                if stride_i == 2 and current_group:
                    # Start a new group at each downsample
                    groups.append(current_group)
                    current_group = []
                    current_stride *= 2

                if s.block_type == "fused":
                    blk = FusedMBConv(
                        last_out_chs, s.out_chs,
                        stride=stride_i, expand_ratio=s.expand_ratio,
                        se_ratio=s.se_ratio, drop_path=dp,
                    )
                else:
                    blk = MBConv(
                        last_out_chs, s.out_chs,
                        stride=stride_i, expand_ratio=s.expand_ratio,
                        kernel_size=s.kernel_size, se_ratio=s.se_ratio, drop_path=dp,
                    )
                current_group.append(blk)
                last_out_chs = s.out_chs
                block_idx += 1

        if current_group:
            groups.append(current_group)

        self.stage_groups = nn.ModuleList(
            nn.Sequential(*g) for g in groups
        )

        # ── Head conv (H/32 resolution, embed_dim channels) ──────────────────
        self.head_conv = nn.Sequential(
            nn.Conv2d(last_out_chs, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.SiLU(inplace=True),
        )
        self.pool      = nn.AdaptiveAvgPool2d(1)
        self.head_drop = nn.Dropout(drop_rate)

        # Feature dims are the output channels of each stride group
        # (roughly strides 4, 8, 16, 32 — may vary by stage config)
        self.feature_dims = [g[-1][-1] if hasattr(g[-1], '__len__') else
                             (list(g[-1].parameters())[0].shape[0]
                              if list(g[-1].parameters()) else 0)
                             for g in groups[:4]]
        # Simpler: just use the last layer's out_channels per group
        self.feature_dims = self._compute_feature_dims(groups)

        self._init_weights()

    def _compute_feature_dims(self, groups) -> list[int]:
        """Get output channel count of each stage group."""
        dims = []
        for g in groups:
            last_block = g[-1]
            # Walk through Sequential to find the last Conv2d or BN
            for m in reversed(list(last_block.modules())):
                if isinstance(m, nn.BatchNorm2d):
                    dims.append(m.num_features)
                    break
        return dims[:4]

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        x = self.stem(x)
        feature_maps: list[torch.Tensor] = []

        for group in self.stage_groups:
            x = group(x)
            feature_maps.append(x)

        # Apply head conv at final scale
        x = self.head_conv(x)
        # Replace last feature map with head-conv output for richer features
        if feature_maps:
            feature_maps[-1] = x

        # Select requested output scales
        indices = self.out_indices if self.out_indices is not None else list(range(len(feature_maps)))
        selected = [feature_maps[i] for i in indices if i < len(feature_maps)]

        pooled = self.pool(x).flatten(1)
        pooled = self.head_drop(pooled)

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=None,
            feature_maps=selected,
        )

    def freeze_up_to_stage(self, stage: int) -> None:
        """Freeze stem and stage groups 0..stage."""
        for p in self.stem.parameters():
            p.requires_grad_(False)
        for i, g in enumerate(self.stage_groups):
            if i <= stage:
                for p in g.parameters():
                    p.requires_grad_(False)


def _make_divisible(v: float, divisor: int = 8) -> int:
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


# ── Stage configs per variant ─────────────────────────────────────────────────

def _s_stages():
    return [
        _StageConfig("fused", 24,  24,  1, 1, 3, 2,  0.0),
        _StageConfig("fused", 24,  48,  2, 4, 3, 4,  0.0),
        _StageConfig("fused", 48,  64,  2, 4, 3, 4,  0.0),
        _StageConfig("mbconv", 64, 128,  2, 4, 3, 6,  0.25),
        _StageConfig("mbconv", 128, 160, 1, 6, 3, 9,  0.25),
        _StageConfig("mbconv", 160, 256, 2, 6, 3, 15, 0.25),
    ]

def _m_stages():
    return [
        _StageConfig("fused", 24,  24,  1, 1, 3, 3,  0.0),
        _StageConfig("fused", 24,  48,  2, 4, 3, 5,  0.0),
        _StageConfig("fused", 48,  80,  2, 4, 3, 5,  0.0),
        _StageConfig("mbconv", 80, 160,  2, 4, 3, 7,  0.25),
        _StageConfig("mbconv", 160, 176, 1, 6, 3, 14, 0.25),
        _StageConfig("mbconv", 176, 304, 2, 6, 3, 18, 0.25),
        _StageConfig("mbconv", 304, 512, 1, 6, 3, 5,  0.25),
    ]

def _l_stages():
    return [
        _StageConfig("fused", 32,  32,  1, 1, 3, 4,  0.0),
        _StageConfig("fused", 32,  64,  2, 4, 3, 7,  0.0),
        _StageConfig("fused", 64,  96,  2, 4, 3, 7,  0.0),
        _StageConfig("mbconv", 96, 192,  2, 4, 3, 10, 0.25),
        _StageConfig("mbconv", 192, 224, 1, 6, 3, 19, 0.25),
        _StageConfig("mbconv", 224, 384, 2, 6, 3, 25, 0.25),
        _StageConfig("mbconv", 384, 640, 1, 6, 3, 7,  0.25),
    ]

def _xl_stages():
    return [
        _StageConfig("fused", 32,  32,  1, 1, 3, 4,  0.0),
        _StageConfig("fused", 32,  64,  2, 4, 3, 8,  0.0),
        _StageConfig("fused", 64,  96,  2, 4, 3, 8,  0.0),
        _StageConfig("mbconv", 96, 192,  2, 4, 3, 16, 0.25),
        _StageConfig("mbconv", 192, 256, 1, 6, 3, 24, 0.25),
        _StageConfig("mbconv", 256, 512, 2, 6, 3, 32, 0.25),
        _StageConfig("mbconv", 512, 640, 1, 6, 3, 8,  0.25),
    ]


# ── Registration ─────────────────────────────────────────────────────────────

@register_vision_backbone("efficientnet_v2_s")
class EfficientNetV2S(EfficientNetV2Backbone):
    """EfficientNetV2-S: ~21M params, fast training with Fused-MBConv."""
    arch_name = "efficientnet_v2_s"
    embed_dim = 1280
    def __init__(self, **kwargs):
        super().__init__(stages=_s_stages(), head_channels=1280, **kwargs)


@register_vision_backbone("efficientnet_v2_m")
class EfficientNetV2M(EfficientNetV2Backbone):
    """EfficientNetV2-M: ~54M params, balanced accuracy/speed."""
    arch_name = "efficientnet_v2_m"
    embed_dim = 1280
    def __init__(self, **kwargs):
        super().__init__(stages=_m_stages(), head_channels=1280, **kwargs)


@register_vision_backbone("efficientnet_v2_l")
class EfficientNetV2L(EfficientNetV2Backbone):
    """EfficientNetV2-L: ~119M params, high accuracy."""
    arch_name = "efficientnet_v2_l"
    embed_dim = 1280
    def __init__(self, **kwargs):
        super().__init__(stages=_l_stages(), head_channels=1280, **kwargs)


@register_vision_backbone("efficientnet_v2_xl")
class EfficientNetV2XL(EfficientNetV2Backbone):
    """EfficientNetV2-XL: ~208M params, largest variant with extended training data."""
    arch_name = "efficientnet_v2_xl"
    embed_dim = 1280
    def __init__(self, **kwargs):
        super().__init__(stages=_xl_stages(), head_channels=1280, **kwargs)
