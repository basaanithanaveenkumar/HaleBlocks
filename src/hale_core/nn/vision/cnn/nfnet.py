"""
NFNet Vision Backbones
======================

Implements the Normalizer-Free Network family (Brock et al. 2021,
"High-Performance Large-Scale Image Recognition Without Normalization",
https://arxiv.org/abs/2102.06171).

Architecture
------------
NFNet achieves state-of-the-art accuracy without BatchNorm via three innovations:

  1. Scaled Weight Standardisation (SWS): normalises convolution weights at
     each forward pass so activation variance stays bounded, replacing BN.

  2. Variance Preservation in Residual Branches: the standard residual formula
     ``y = x + F(x)`` grows activation variance monotonically.  NFNet corrects
     this with ``y = x + alpha * F(x / beta)`` where alpha ≈ 0.2 (learned) and
     beta = sqrt(accumulated variance), keeping Var(y) ≈ Var(x).

  3. Adaptive Gradient Clipping (AGC): clips gradients by the ratio of the
     parameter norm to the gradient norm, ``g ← g * min(1, λ ||w|| / ||g||)``.
     This is a training-time utility; see ``agc_clip()`` below.

Block structure
---------------
  NFBlock (SE variant, group conv):
    WSConv 1×1 (pre-activation, /beta)
    → GELU
    → WSConv 3×3 groups=group_size (stride if downsample)
    → GELU
    → WSConv 1×1
    → SE (squeeze-excite, squeeze_ratio=0.5)
    → * alpha (learnable)
    → + residual (projected shortcut if needed)

Stage configs (widths, depths, group_size=128):
  F0: depths=[1,2,6,3],   widths=[256,512,1536,1536]
  F1: depths=[2,4,12,6],  widths=[256,512,1536,1536]
  F2: depths=[3,6,18,9],  widths=[256,512,1536,1536]
  F3: depths=[4,8,24,12], widths=[256,512,1536,1536]
  F4: depths=[5,10,30,15],widths=[256,512,1536,1536]
  F5: depths=[6,12,36,18],widths=[256,512,1536,1536]

Stem: 4 WSConv2d layers totalling 4× spatial reduction.

Registered names
----------------
  nfnet_f0 — depths=[1,2,6,3]
  nfnet_f1 — depths=[2,4,12,6]
  nfnet_f2 — depths=[3,6,18,9]
  nfnet_f3 — depths=[4,8,24,12]
  nfnet_f4 — depths=[5,10,30,15]
  nfnet_f5 — depths=[6,12,36,18]

Citation
--------
  Brock, A., De, S., Smith, S. L., & Simonyan, K. (2021).
  High-Performance Large-Scale Image Recognition Without Normalization. ICML.
  arXiv:2102.06171
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.registry import register_vision_backbone


# ── Adaptive Gradient Clipping ────────────────────────────────────────────────

def agc_clip(model: nn.Module, lambda_: float = 0.01) -> None:
    """Adaptive Gradient Clipping: clips each parameter's gradient by its weight norm.

    Per-parameter clipping: g ← g * min(1, λ * ||w|| / ||g||).
    Should be called after loss.backward() before optimizer.step().

    Args:
        model:   The NFNet model (or any nn.Module with gradients).
        lambda_: Clipping coefficient (default 0.01, as in the paper).
    """
    for p in model.parameters():
        if p.grad is None:
            continue
        # Per-parameter weight and gradient norms (across all dims except batch)
        param_norm = p.data.norm(dim=list(range(1, p.data.ndim)), keepdim=True).clamp(min=1e-3)
        grad_norm  = p.grad.norm(dim=list(range(1, p.grad.ndim)), keepdim=True)
        max_norm   = param_norm * lambda_
        clip_coef  = (max_norm / grad_norm.clamp(min=1e-6)).clamp(max=1.0)
        p.grad.mul_(clip_coef)


# ── Scaled Weight Standardisation conv ───────────────────────────────────────

class WSConv2d(nn.Conv2d):
    """Conv2d with Scaled Weight Standardisation (Brock et al. 2021, Sec. 3.1).

    Normalises filter weights at forward time so that each conv output has
    approximately unit variance, eliminating the need for BatchNorm.

    A learnable per-output-channel gain ``self.gain`` scales the normalised
    weights to preserve the expected signal variance under GELU activations.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Per-output-channel gain (replaces BN's learnable scale)
        self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        # Subtract per-filter mean (dims: in_C, kH, kW)
        w = w - w.mean(dim=[1, 2, 3], keepdim=True)
        # Normalise variance — eps avoids div-by-zero for zero-weight filters
        w = w / (w.var(dim=[1, 2, 3], keepdim=True, unbiased=False) + 1e-5).sqrt()
        # Scale to maintain expected variance: gain / sqrt(fan_in)
        fan_in = w[0].numel()
        w = w * (self.gain / math.sqrt(fan_in))
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


# ── Squeeze-Excite ────────────────────────────────────────────────────────────

class SqueezeExcite(nn.Module):
    """Channel attention: global-pool → squeeze linear → GELU → excite linear → sigmoid."""

    def __init__(self, in_channels: int, squeeze_ratio: float = 0.5) -> None:
        super().__init__()
        mid = max(1, int(in_channels * squeeze_ratio))
        self.squeeze = nn.Linear(in_channels, mid)
        self.excite  = nn.Linear(mid, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        scale = x.mean(dim=[2, 3])           # [B, C] global average pool
        scale = F.gelu(self.squeeze(scale))  # [B, mid]
        scale = torch.sigmoid(self.excite(scale))  # [B, C]
        return x * scale.unsqueeze(-1).unsqueeze(-1)


# ── Stochastic depth ──────────────────────────────────────────────────────────

class StochasticDepth(nn.Module):
    """Randomly drops the entire block output during training."""

    def __init__(self, drop_prob: float) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.floor(torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob)
        return x * noise / keep_prob


# ── NFBlock ───────────────────────────────────────────────────────────────────

class NFBlock(nn.Module):
    """Normalizer-Free residual block with variance preservation.

    Uses the formula: y = x + alpha * F(x / beta)
    where alpha (scalar, ≈0.2) scales the residual branch to prevent variance
    accumulation, and beta = expected_std_of_x normalises the input.

    Args:
        in_channels:   Input channel count.
        out_channels:  Output channel count.
        stride:        Spatial stride for the 3×3 conv.
        group_size:    Group width for grouped conv.
        beta:          Expected input standard deviation (accumulated across blocks).
        alpha:         Residual branch scale factor.
        squeeze_ratio: SE squeeze ratio (0 = no SE).
        drop_path:     Stochastic depth rate.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        stride:       int   = 1,
        group_size:   int   = 128,
        beta:         float = 1.0,
        alpha:        float = 0.2,
        squeeze_ratio: float = 0.5,
        drop_path:    float = 0.0,
    ) -> None:
        super().__init__()
        self.beta  = beta
        # Learnable alpha per block (init to provided alpha value)
        self.alpha = nn.Parameter(torch.full([], alpha))

        groups = max(1, out_channels // group_size)
        # Bottleneck: in → out → out (groups) → out
        self.conv1 = WSConv2d(in_channels,  out_channels, 1, bias=False)
        self.conv2 = WSConv2d(out_channels, out_channels, 3, stride=stride,
                              padding=1, groups=groups, bias=False)
        self.conv3 = WSConv2d(out_channels, out_channels, 1, bias=False)

        self.se         = SqueezeExcite(out_channels, squeeze_ratio) if squeeze_ratio > 0 else nn.Identity()
        self.drop_path  = StochasticDepth(drop_path) if drop_path > 0 else nn.Identity()

        # Skip connection if shape changes
        self.shortcut: Optional[nn.Module] = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = WSConv2d(in_channels, out_channels, 1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise input variance before the residual branch
        x_norm = x / self.beta

        out = F.gelu(self.conv1(x_norm))
        out = F.gelu(self.conv2(out))
        out = self.conv3(out)
        out = self.se(out)
        out = self.drop_path(out) * self.alpha

        skip = x if self.shortcut is None else self.shortcut(x)
        return out + skip


# ── NFNet backbone ────────────────────────────────────────────────────────────

_NFNET_WIDTHS = [256, 512, 1536, 1536]
_NFNET_DEPTHS = {
    "f0": [1,  2,  6,  3],
    "f1": [2,  4, 12,  6],
    "f2": [3,  6, 18,  9],
    "f3": [4,  8, 24, 12],
    "f4": [5, 10, 30, 15],
    "f5": [6, 12, 36, 18],
}


class NFNet(BaseVisionBackbone):
    """NFNet: Normalizer-Free high-performance backbone.

    No BatchNorm anywhere.  Weight standardisation + variance preservation
    residuals + adaptive gradient clipping at training time.

    Args:
        variant:        One of 'f0'..'f5' controlling depth multipliers.
        drop_path_rate: Max stochastic depth rate (linearly distributed).
        img_size:       Input spatial resolution.
        in_chans:       Input channels (default 3).
        drop_rate:      Dropout after global pool.
        out_indices:    Which stage feature maps to return.
    """

    embed_dim    = 1536
    feature_dims = [256, 512, 1536, 1536]

    def __init__(
        self,
        variant:         str   = "f0",
        drop_path_rate:  float = 0.0,
        img_size:        int   = 224,
        in_chans:        int   = 3,
        drop_rate:       float = 0.0,
        out_indices:     list[int] | None = None,
    ) -> None:
        super().__init__(img_size=img_size, in_chans=in_chans, drop_rate=drop_rate, out_indices=out_indices)

        depths = _NFNET_DEPTHS[variant]
        widths = _NFNET_WIDTHS

        # ── Stem: 4 WSConv layers → 4× spatial reduction ─────────────────────
        # Each stride-2 conv halves spatial dims; two stride-2 = 4× total.
        self.stem = nn.Sequential(
            WSConv2d(in_chans, 16, 3, stride=2, padding=1, bias=False),
            nn.GELU(),
            WSConv2d(16, 32, 3, stride=1, padding=1, bias=False),
            nn.GELU(),
            WSConv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.GELU(),
            WSConv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.GELU(),
        )

        # ── Build 4 stages ────────────────────────────────────────────────────
        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        self.stages = nn.ModuleList()
        block_idx = 0
        # Tracks expected std for variance-preservation beta calculation
        expected_std = 1.0
        in_ch = 128

        for stage_idx, (depth, width) in enumerate(zip(depths, widths)):
            blocks = []
            for i in range(depth):
                stride = 2 if (i == 0 and stage_idx > 0) else 1
                # beta = current expected std; resets to 1 if this block downsamples
                beta = expected_std
                blocks.append(NFBlock(
                    in_channels=in_ch,
                    out_channels=width,
                    stride=stride,
                    group_size=128,
                    beta=beta,
                    alpha=0.2,
                    squeeze_ratio=0.5,
                    drop_path=dp_rates[block_idx],
                ))
                in_ch = width
                block_idx += 1
                # Variance accumulates across blocks: std grows by sqrt(1 + alpha^2)
                expected_std = math.sqrt(expected_std ** 2 + 0.2 ** 2)
                if stride > 1:
                    expected_std = 1.0  # reset after downsampling

            self.stages.append(nn.Sequential(*blocks))

        # ── Head ─────────────────────────────────────────────────────────────
        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.head_fc = nn.Linear(widths[-1], self.embed_dim)
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """NFNet forward: stem → 4 stages → pool → head.

        Returns:
            BackboneOutput with pooled_output [B, 1536] and 4 feature maps.
        """
        x = self.stem(x)  # [B, 128, H/4, W/4]

        feature_maps = []
        for stage in self.stages:
            x = stage(x)
            feature_maps.append(x)

        pooled = self.pool(feature_maps[-1]).flatten(1)  # [B, 1536]
        pooled = F.gelu(pooled)
        pooled = self.dropout(self.head_fc(pooled))

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


# ── Registered variants ───────────────────────────────────────────────────────

@register_vision_backbone("nfnet_f0")
class NFNetF0(NFNet):
    """NFNet-F0: depths=[1,2,6,3] — smallest, 71M params."""
    arch_name    = "nfnet_f0"
    embed_dim    = 1536
    feature_dims = [256, 512, 1536, 1536]

    def __init__(self, **kwargs) -> None:
        super().__init__(variant="f0", **kwargs)


@register_vision_backbone("nfnet_f1")
class NFNetF1(NFNet):
    """NFNet-F1: depths=[2,4,12,6] — 132M params."""
    arch_name    = "nfnet_f1"
    embed_dim    = 1536
    feature_dims = [256, 512, 1536, 1536]

    def __init__(self, **kwargs) -> None:
        super().__init__(variant="f1", **kwargs)


@register_vision_backbone("nfnet_f2")
class NFNetF2(NFNet):
    """NFNet-F2: depths=[3,6,18,9] — 194M params."""
    arch_name    = "nfnet_f2"
    embed_dim    = 1536
    feature_dims = [256, 512, 1536, 1536]

    def __init__(self, **kwargs) -> None:
        super().__init__(variant="f2", **kwargs)


@register_vision_backbone("nfnet_f3")
class NFNetF3(NFNet):
    """NFNet-F3: depths=[4,8,24,12] — 255M params."""
    arch_name    = "nfnet_f3"
    embed_dim    = 1536
    feature_dims = [256, 512, 1536, 1536]

    def __init__(self, **kwargs) -> None:
        super().__init__(variant="f3", **kwargs)


@register_vision_backbone("nfnet_f4")
class NFNetF4(NFNet):
    """NFNet-F4: depths=[5,10,30,15] — 316M params."""
    arch_name    = "nfnet_f4"
    embed_dim    = 1536
    feature_dims = [256, 512, 1536, 1536]

    def __init__(self, **kwargs) -> None:
        super().__init__(variant="f4", **kwargs)


@register_vision_backbone("nfnet_f5")
class NFNetF5(NFNet):
    """NFNet-F5: depths=[6,12,36,18] — largest, 377M params, 89.2% top-1."""
    arch_name    = "nfnet_f5"
    embed_dim    = 1536
    feature_dims = [256, 512, 1536, 1536]

    def __init__(self, **kwargs) -> None:
        super().__init__(variant="f5", **kwargs)
