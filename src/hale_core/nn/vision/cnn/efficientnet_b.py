"""
EfficientNet B0-B7 Vision Backbones
======================================

Implements the original EfficientNet family (Tan & Le 2019, "EfficientNet:
Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019).

Reference: https://arxiv.org/abs/1905.11946

Architecture
------------
EfficientNet introduces **compound scaling**: width, depth and input resolution
are scaled simultaneously using a single compound coefficient φ:

    depth    d = α^φ
    width    w = β^φ
    resolution r = γ^φ

Subject to: α × β² × γ² ≈ 2, α≥1, β≥1, γ≥1.
B0 baseline found via NAS; B1-B7 apply compound scaling.

Building block: **MBConv** (Mobile Inverted Bottleneck Convolution) with SE:
  input → expand 1×1 (ratio r) → BN → SiLU
        → depthwise k×k        → BN → SiLU
        → SE(reduction=4)
        → project 1×1          → BN
        → (+residual if stride=1 and in_chs==out_chs)

B0 baseline stage layout (Table 1 of paper):
  Stage 1 : MBConv1, k=3, s=1, out=16,  repeat=1
  Stage 2 : MBConv6, k=3, s=2, out=24,  repeat=2
  Stage 3 : MBConv6, k=5, s=2, out=40,  repeat=2
  Stage 4 : MBConv6, k=3, s=2, out=80,  repeat=3
  Stage 5 : MBConv6, k=5, s=1, out=112, repeat=3
  Stage 6 : MBConv6, k=5, s=2, out=192, repeat=4
  Stage 7 : MBConv6, k=3, s=1, out=320, repeat=1
  Head    : Conv 1×1 → 1280 → GlobalAvgPool → dropout → embedding

Multi-scale feature map outputs:
  feature_maps[0]: stride-4  (after stem)
  feature_maps[1]: stride-8  (after stage 2)
  feature_maps[2]: stride-16 (after stage 4)
  feature_maps[3]: stride-32 (after stage 6+7+head)

Compound scaling multipliers (from paper Table 2):
  B0: depth=1.0, width=1.0, res=224
  B1: depth=1.1, width=1.0, res=240
  B2: depth=1.2, width=1.1, res=260
  B3: depth=1.4, width=1.2, res=300
  B4: depth=1.8, width=1.4, res=380
  B5: depth=2.2, width=1.6, res=456
  B6: depth=2.6, width=1.8, res=528
  B7: depth=3.1, width=2.0, res=600

Registered names
----------------
  efficientnet_b0  — ~5.3M params
  efficientnet_b1  — ~7.8M params
  efficientnet_b2  — ~9.1M params
  efficientnet_b3  — ~12M params
  efficientnet_b4  — ~19M params
  efficientnet_b5  — ~30M params
  efficientnet_b6  — ~43M params
  efficientnet_b7  — ~66M params

Note: This is the ORIGINAL EfficientNet (B-series).  EfficientNetV2 is a
separate architecture in efficientnet.py (uses FusedMBConv blocks).

Citation
--------
  Tan, M. & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for
  Convolutional Neural Networks. ICML. arXiv:1905.11946.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.registry import register_vision_backbone


# ── Utilities ─────────────────────────────────────────────────────────────────

def _make_divisible(v: float, divisor: int = 8) -> int:
    """Round v up to the nearest multiple of divisor (min=divisor)."""
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


# ── Stochastic Depth ──────────────────────────────────────────────────────────

class StochasticDepth(nn.Module):
    """Randomly drops residual branches during training (Huang et al. 2016).

    At test time, the residual branch is always active (no scaling needed;
    the expected value is preserved by the Bernoulli / survival probability).
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        survival = 1.0 - self.drop_prob
        noise = torch.empty(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype)
        noise.bernoulli_(survival).div_(survival)
        return x * noise


# ── Squeeze-Excitation (SE ratio = 1/4 of expanded channels, not input) ────

class SqueezeExciteB(nn.Module):
    """SE block for EfficientNet-B with SiLU activations.

    Args:
        in_chs:       Input (and output) channel count.
        se_ratio:     Reduction ratio applied to in_chs.  Default 0.25 (paper: 0.25).
    """

    def __init__(self, in_chs: int, se_ratio: float = 0.25) -> None:
        super().__init__()
        squeezed = max(1, int(in_chs * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1  = nn.Conv2d(in_chs, squeezed, kernel_size=1, bias=True)
        self.fc2  = nn.Conv2d(squeezed, in_chs, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = self.pool(x)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


# ── MBConv block ──────────────────────────────────────────────────────────────

class MBConvB(nn.Module):
    """Mobile Inverted Bottleneck Conv for EfficientNet-B series.

    When expand_ratio=1 the expansion 1×1 conv is skipped (no channel change).
    SE is applied to the expanded (depthwise) channel count, not the input count.
    This differs from SE-ResNet where SE is applied to the bottleneck output.

    Args:
        in_chs:       Input channels.
        out_chs:      Output channels.
        stride:       Spatial stride (1 or 2).
        expand_ratio: Channel expansion before depthwise (typically 6, or 1 for first stage).
        kernel_size:  Depthwise conv kernel size (3 or 5).
        se_ratio:     SE reduction ratio relative to expanded channels.
        drop_path:    Stochastic depth probability.
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
        pad      = (kernel_size - 1) // 2

        layers: list[nn.Module] = []

        # Expansion phase (skip when ratio=1 — nothing changes channel-wise)
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_chs, mid_chs, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_chs),
                nn.SiLU(inplace=True),
            ]
        else:
            mid_chs = in_chs   # no expansion

        # Depthwise conv
        layers += [
            nn.Conv2d(mid_chs, mid_chs, kernel_size=kernel_size,
                      stride=stride, padding=pad, groups=mid_chs, bias=False),
            nn.BatchNorm2d(mid_chs),
            nn.SiLU(inplace=True),
        ]

        # Squeeze-Excite on expanded channels
        if se_ratio > 0.0:
            layers.append(SqueezeExciteB(mid_chs, se_ratio=se_ratio))

        # Projection
        layers += [
            nn.Conv2d(mid_chs, out_chs, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_chs),
        ]

        self.conv      = nn.Sequential(*layers)
        self.drop_path = StochasticDepth(drop_path) if has_skip else nn.Identity()
        self.has_skip  = has_skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_skip:
            return x + self.drop_path(self.conv(x))
        return self.conv(x)


# ── Stage configuration ───────────────────────────────────────────────────────

@dataclass
class _B0StageConfig:
    """B0 baseline specification for one stage."""
    expand_ratio: int
    kernel_size:  int
    stride:       int
    out_chs:      int   # baseline (B0) output channels
    num_blocks:   int   # baseline (B0) repeat count


# B0 stage definitions from Table 1 of the EfficientNet paper
_B0_STAGES: list[_B0StageConfig] = [
    _B0StageConfig(1, 3, 1, 16,  1),  # Stage 1: MBConv1
    _B0StageConfig(6, 3, 2, 24,  2),  # Stage 2: MBConv6
    _B0StageConfig(6, 5, 2, 40,  2),  # Stage 3: MBConv6 k=5
    _B0StageConfig(6, 3, 2, 80,  3),  # Stage 4: MBConv6
    _B0StageConfig(6, 5, 1, 112, 3),  # Stage 5: MBConv6 k=5
    _B0StageConfig(6, 5, 2, 192, 4),  # Stage 6: MBConv6 k=5
    _B0StageConfig(6, 3, 1, 320, 1),  # Stage 7: MBConv6
]


# ── Core EfficientNet-B model ─────────────────────────────────────────────────

class EfficientNetB(BaseVisionBackbone):
    """Original EfficientNet B0-B7 backbone.

    Scales the B0 baseline architecture using compound depth/width multipliers.
    Returns a 4-scale feature pyramid compatible with FPN/BEV heads.

    The feature pyramid groups stages by their spatial stride:
      feature_maps[0] → stride 4  (stem output)
      feature_maps[1] → stride 8  (after first stride-2 stage)
      feature_maps[2] → stride 16 (after second stride-2 stage)
      feature_maps[3] → stride 32 (after third stride-2 stage + remaining stages + head)

    Args:
        depth_mult:   Depth scaling factor (multiplies num_blocks per stage).
        width_mult:   Width scaling factor (multiplies channel counts).
        head_chs:     Head 1×1 conv output channels (default 1280).
        drop_path_rate: Max stochastic depth rate.
        dropout_rate: Classifier dropout (applied to pooled embedding).
        img_size, in_chans, drop_rate, out_indices: See BaseVisionBackbone.
    """

    arch_name:    str       = "efficientnet_b"
    embed_dim:    int       = 1280
    feature_dims: list[int] = []

    def __init__(
        self,
        depth_mult:     float = 1.0,
        width_mult:     float = 1.0,
        head_chs:       int   = 1280,
        drop_path_rate: float = 0.2,
        img_size:       int   = 224,
        in_chans:       int   = 3,
        drop_rate:      float = 0.2,
        out_indices:    list[int] | None = None,
    ) -> None:
        self.embed_dim = head_chs
        super().__init__(img_size=img_size, in_chans=in_chans,
                         drop_rate=drop_rate, out_indices=out_indices)

        def _w(c: int) -> int:
            return _make_divisible(c * width_mult, 8)

        def _d(n: int) -> int:
            return max(1, math.ceil(n * depth_mult))

        # ── Stem: stride-2, 3→stem_chs ───────────────────────────────────────
        stem_chs = _w(32)
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, stem_chs, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_chs),
            nn.SiLU(inplace=True),
        )

        # ── Build scaled stages ───────────────────────────────────────────────
        total_blocks = sum(_d(s.num_blocks) for s in _B0_STAGES)
        block_idx    = 0

        # Group blocks by feature map scale (stride-2 transitions)
        # Groups: [stem→s1], [s2], [s3], [s4+s5+s6+s7+head]
        # stride-2 stages: 2 (24ch), 3 (40ch), 4 (80ch), 6 (192ch)
        # We track stride to assign blocks to feature pyramid levels.
        stride_groups: list[list[nn.Module]] = [[] for _ in range(4)]
        current_group = 0
        stride_counts = [0]   # cumulative stride-doublings seen so far

        in_chs = stem_chs
        stride_2_seen = 0

        stage_modules: list[nn.Sequential] = []

        for s in _B0_STAGES:
            out_chs    = _w(s.out_chs)
            num_blocks = _d(s.num_blocks)
            stage_blks: list[nn.Module] = []

            for i in range(num_blocks):
                dp   = drop_path_rate * block_idx / max(total_blocks - 1, 1)
                blk  = MBConvB(
                    in_chs=in_chs,
                    out_chs=out_chs,
                    stride=s.stride if i == 0 else 1,
                    expand_ratio=s.expand_ratio,
                    kernel_size=s.kernel_size,
                    se_ratio=0.25,
                    drop_path=dp,
                )
                stage_blks.append(blk)
                in_chs = out_chs
                block_idx += 1

            stage_modules.append(nn.Sequential(*stage_blks))
            if s.stride == 2:
                stride_2_seen += 1

        self.stages = nn.ModuleList(stage_modules)

        # ── Head conv ─────────────────────────────────────────────────────────
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_chs, head_chs, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_chs),
            nn.SiLU(inplace=True),
        )
        self.pool      = nn.AdaptiveAvgPool2d(1)
        self.head_drop = nn.Dropout(drop_rate)

        # Feature dims: output channels of stride-2 transition stages + head
        # Stages with stride=2: indices 1, 2, 3, 5 in _B0_STAGES (0-indexed)
        stride2_stages = [i for i, s in enumerate(_B0_STAGES) if s.stride == 2]
        # feature_dims[0] = channels after stem (stride-4)
        # feature_dims[1..3] = channels after each of the first 3 stride-2 stages
        stride2_chs = [_w(_B0_STAGES[i].out_chs) for i in stride2_stages[:3]]
        self.feature_dims = [stem_chs] + stride2_chs + [head_chs]
        # Trim to 4 levels
        self.feature_dims = self.feature_dims[:4]

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
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """Run EfficientNet-B forward pass.

        Args:
            x: [B, in_chans, H, W]

        Returns:
            BackboneOutput:
              pooled_output — [B, head_chs]
              feature_maps  — 4-scale pyramid
        """
        # Stem output: stride-4 (first feature map level)
        x = self.stem(x)
        feature_maps: list[torch.Tensor] = [x]

        stride_count = 0
        for stage in self.stages:
            x = stage(x)
            # Detect stride-2 stages by checking the first block's stride
            first_block = stage[0] if len(stage) > 0 else None
            if first_block is not None:
                # Check if first block had stride=2 (has no skip connection)
                if not getattr(first_block, 'has_skip', True):
                    # This was a stride-2 block — emit feature map
                    if stride_count < 3:
                        feature_maps.append(x)
                    stride_count += 1

        # Apply head conv and use its output for the final feature scale
        x = self.head_conv(x)
        # Ensure we have exactly 4 feature maps: stem + 3 stride-2 transitions
        while len(feature_maps) < 4:
            feature_maps.append(x)
        feature_maps[-1] = x   # last map always uses head output

        indices = self.out_indices if self.out_indices is not None else list(range(len(feature_maps)))
        selected = [feature_maps[i] for i in indices if i < len(feature_maps)]

        pooled = self.head_drop(self.pool(x).flatten(1))

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=None,
            feature_maps=selected,
            attentions=None,
        )

    def freeze_up_to_stage(self, stage: int) -> None:
        """Freeze stem + stages 0..stage of the stage list."""
        for p in self.stem.parameters():
            p.requires_grad_(False)
        for i, s in enumerate(self.stages):
            if i <= stage:
                for p in s.parameters():
                    p.requires_grad_(False)


# ── Compound scaling parameters (from Table 2, EfficientNet paper) ───────────

# (depth_mult, width_mult, default_res, dropout)
_SCALES = {
    "b0": (1.0, 1.0, 224, 0.2),
    "b1": (1.1, 1.0, 240, 0.2),
    "b2": (1.2, 1.1, 260, 0.3),
    "b3": (1.4, 1.2, 300, 0.3),
    "b4": (1.8, 1.4, 380, 0.4),
    "b5": (2.2, 1.6, 456, 0.4),
    "b6": (2.6, 1.8, 528, 0.5),
    "b7": (3.1, 2.0, 600, 0.5),
}


def _build_b(variant: str, **kwargs):
    dm, wm, _, dr = _SCALES[variant]
    kwargs.setdefault("drop_rate", dr)
    return EfficientNetB(depth_mult=dm, width_mult=wm, head_chs=_make_divisible(1280 * wm, 8), **kwargs)


# ── Registered variants ───────────────────────────────────────────────────────

@register_vision_backbone("efficientnet_b0")
class EfficientNetB0(EfficientNetB):
    """EfficientNet-B0 — NAS baseline, ~5.3M params, embed_dim=1280."""
    arch_name = "efficientnet_b0"
    embed_dim = 1280
    def __init__(self, **kwargs):
        dm, wm, _, dr = _SCALES["b0"]
        kwargs.setdefault("drop_rate", dr)
        super().__init__(depth_mult=dm, width_mult=wm, head_chs=1280, **kwargs)


@register_vision_backbone("efficientnet_b1")
class EfficientNetB1(EfficientNetB):
    """EfficientNet-B1 — depth scaled, ~7.8M params, embed_dim=1280."""
    arch_name = "efficientnet_b1"
    embed_dim = 1280
    def __init__(self, **kwargs):
        dm, wm, _, dr = _SCALES["b1"]
        kwargs.setdefault("drop_rate", dr)
        super().__init__(depth_mult=dm, width_mult=wm, head_chs=1280, **kwargs)


@register_vision_backbone("efficientnet_b2")
class EfficientNetB2(EfficientNetB):
    """EfficientNet-B2 — depth+width scaled, ~9.1M params, embed_dim=1408."""
    arch_name = "efficientnet_b2"
    embed_dim = _make_divisible(1280 * 1.1, 8)
    def __init__(self, **kwargs):
        dm, wm, _, dr = _SCALES["b2"]
        kwargs.setdefault("drop_rate", dr)
        super().__init__(depth_mult=dm, width_mult=wm, head_chs=_make_divisible(1280 * wm, 8), **kwargs)


@register_vision_backbone("efficientnet_b3")
class EfficientNetB3(EfficientNetB):
    """EfficientNet-B3 — ~12M params, embed_dim=1536."""
    arch_name = "efficientnet_b3"
    embed_dim = _make_divisible(1280 * 1.2, 8)
    def __init__(self, **kwargs):
        dm, wm, _, dr = _SCALES["b3"]
        kwargs.setdefault("drop_rate", dr)
        super().__init__(depth_mult=dm, width_mult=wm, head_chs=_make_divisible(1280 * wm, 8), **kwargs)


@register_vision_backbone("efficientnet_b4")
class EfficientNetB4(EfficientNetB):
    """EfficientNet-B4 — ~19M params, embed_dim=1792."""
    arch_name = "efficientnet_b4"
    embed_dim = _make_divisible(1280 * 1.4, 8)
    def __init__(self, **kwargs):
        dm, wm, _, dr = _SCALES["b4"]
        kwargs.setdefault("drop_rate", dr)
        super().__init__(depth_mult=dm, width_mult=wm, head_chs=_make_divisible(1280 * wm, 8), **kwargs)


@register_vision_backbone("efficientnet_b5")
class EfficientNetB5(EfficientNetB):
    """EfficientNet-B5 — ~30M params, embed_dim=2048."""
    arch_name = "efficientnet_b5"
    embed_dim = _make_divisible(1280 * 1.6, 8)
    def __init__(self, **kwargs):
        dm, wm, _, dr = _SCALES["b5"]
        kwargs.setdefault("drop_rate", dr)
        super().__init__(depth_mult=dm, width_mult=wm, head_chs=_make_divisible(1280 * wm, 8), **kwargs)


@register_vision_backbone("efficientnet_b6")
class EfficientNetB6(EfficientNetB):
    """EfficientNet-B6 — ~43M params, embed_dim=2304."""
    arch_name = "efficientnet_b6"
    embed_dim = _make_divisible(1280 * 1.8, 8)
    def __init__(self, **kwargs):
        dm, wm, _, dr = _SCALES["b6"]
        kwargs.setdefault("drop_rate", dr)
        super().__init__(depth_mult=dm, width_mult=wm, head_chs=_make_divisible(1280 * wm, 8), **kwargs)


@register_vision_backbone("efficientnet_b7")
class EfficientNetB7(EfficientNetB):
    """EfficientNet-B7 — ~66M params, embed_dim=2560."""
    arch_name = "efficientnet_b7"
    embed_dim = _make_divisible(1280 * 2.0, 8)
    def __init__(self, **kwargs):
        dm, wm, _, dr = _SCALES["b7"]
        kwargs.setdefault("drop_rate", dr)
        super().__init__(depth_mult=dm, width_mult=wm, head_chs=_make_divisible(1280 * wm, 8), **kwargs)
