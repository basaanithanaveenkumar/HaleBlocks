"""
MobileOne — Re-Parameterisable Efficient Backbone (Vasu et al. 2022, Apple Research)
======================================================================================

Reference: "MobileOne: An Improved One millisecond Mobile Backbone"
           https://arxiv.org/abs/2206.04040

Key Idea: Structural Re-Parameterisation
-----------------------------------------
At TRAINING time, each MobileOne block uses MULTIPLE parallel branches:
  - k branches of depthwise 3×3 conv (each with BN)
  - k branches of 1×1 pointwise conv (each with BN)
  - 1 identity branch (BN only, if in_channels == out_channels and stride == 1)

At INFERENCE time, all branches are FUSED into a SINGLE 3×3 conv + bias,
achieving the same output as all branches combined — but with zero overhead.

This gives the expressiveness of multi-branch training with the efficiency
of single-conv inference.

Reparameterisation formula:
  For each branch (weight W_i, BN: γ, β, μ, σ²), the equivalent conv is:
    W_eq = γ/σ · W_i
    b_eq = β - γ·μ/σ
  All branches are summed: W = Σ W_eq_i, b = Σ b_eq_i

  For the identity branch: W is a 1-in-1-out identity kernel (eye for depthwise,
  pointwise identity), then merged as above.

Why this works:
  BN(Conv(x)) = γ/σ · Conv(x) + (β - γμ/σ)
  Sum of linear ops (BN∘Conv) with same input = single linear op.
  So the entire k-branch structure can be collapsed to one conv.

Variants (s0..s4)
-----------------
  s0: width_multiplier=[0.75, 1.0, 1.0, 2.0], k=4  — smallest, fast on edge
  s1: width_multiplier=[1.5,  1.5, 2.0, 2.5], k=4  — balanced
  s2: width_multiplier=[1.5,  2.0, 2.5, 4.0], k=4  — higher accuracy
  s3: width_multiplier=[2.0,  2.5, 3.0, 4.0], k=4  — large
  s4: width_multiplier=[3.0,  3.5, 3.5, 4.0], k=4  — largest

After reparametrize(), model is in inference mode (single conv per block).
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.registry import register_vision_backbone


# ── Reparametrisation Helpers ─────────────────────────────────────────────────

def _fuse_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge Conv + BN into a single conv weight + bias tensor.

    Formula:
        W_fused = (γ / sqrt(σ² + ε)) * W_conv
        b_fused = β - γ * μ / sqrt(σ² + ε)

    Args:
        conv: Conv2d (may have bias=False).
        bn:   BatchNorm2d paired with that conv.

    Returns:
        (weight, bias) both as Tensor.
    """
    w    = conv.weight                                     # [C_out, C_in/G, kH, kW]
    mean = bn.running_mean                                 # [C_out]
    var  = bn.running_var                                  # [C_out]
    beta = bn.bias                                         # [C_out]
    gamma = bn.weight                                      # [C_out]
    eps  = bn.eps

    scale = gamma / (var + eps).sqrt()                     # [C_out]
    # reshape scale for broadcasting over spatial dims
    w_fused = w * scale.view(-1, 1, 1, 1)
    b_fused = beta - mean * scale
    return w_fused, b_fused


# ── Building Blocks ───────────────────────────────────────────────────────────

class _ConvBN(nn.Sequential):
    """Conv2d → BatchNorm2d (no activation)."""

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        stride:       int  = 1,
        groups:       int  = 1,
    ) -> None:
        pad = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=pad, groups=groups, bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )


class MobileOneBlock(nn.Module):
    """A single MobileOne block with structural re-parameterisation.

    In TRAINING mode: uses k parallel branches per conv.
    In INFERENCE mode (after reparametrize()): single fused conv.

    Architecture:
        input
          ↓ Depthwise (groups=in_channels) with k 3×3 branches
          ↓ + identity branch (if stride=1, same C)
        intermediate
          ↓ Pointwise (1×1) with k branches
        output
        (activation applied after both stages by the caller)

    Args:
        in_channels:  Input channel count.
        out_channels: Output channel count.
        stride:       Spatial stride (1 for same-size, 2 for downsampling).
        num_blocks:   k — number of parallel conv branches per stage.
                      k=1 gives the inference model; k>1 for rich training.
        use_act:      Whether to apply ReLU after this block.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        stride:       int  = 1,
        num_blocks:   int  = 1,
        use_act:      bool = True,
    ) -> None:
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.stride       = stride
        self.num_blocks   = num_blocks
        self.use_act      = use_act
        self._is_reparam  = False  # True after reparametrize()

        # ── Depthwise convolution branches ───────────────────────────────────
        self.dw_branches: nn.ModuleList = nn.ModuleList([
            _ConvBN(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels)
            for _ in range(num_blocks)
        ])
        # 1×1 branch in depthwise (acts as identity-like for same-size kernels)
        self.dw_skip_1x1: Optional[_ConvBN] = None
        if in_channels == out_channels:
            self.dw_skip_1x1 = _ConvBN(
                in_channels, in_channels, kernel_size=1, stride=stride, groups=in_channels
            )
        # Identity branch for depthwise (only if stride=1 and same channels)
        self.dw_bn: Optional[nn.BatchNorm2d] = None
        if stride == 1 and in_channels == out_channels:
            self.dw_bn = nn.BatchNorm2d(in_channels)

        # ── Pointwise convolution branches ───────────────────────────────────
        self.pw_branches: nn.ModuleList = nn.ModuleList([
            _ConvBN(in_channels, out_channels, kernel_size=1)
            for _ in range(num_blocks)
        ])
        self.pw_bn: Optional[nn.BatchNorm2d] = None
        if in_channels == out_channels:
            # Identity branch for pointwise
            self.pw_bn = nn.BatchNorm2d(in_channels)

        # ── Activation ───────────────────────────────────────────────────────
        self.act = nn.ReLU() if use_act else nn.Identity()

        # ── Reparam'd single conv (filled by reparametrize()) ─────────────────
        self.reparam_dw: Optional[nn.Conv2d] = None
        self.reparam_pw: Optional[nn.Conv2d] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._is_reparam:
            # Fast inference path: single fused conv
            return self.act(self.reparam_pw(self.act(self.reparam_dw(x))))

        # ── Depthwise stage ───────────────────────────────────────────────────
        dw_out = sum(branch(x) for branch in self.dw_branches)
        if self.dw_skip_1x1 is not None:
            dw_out = dw_out + self.dw_skip_1x1(x)
        if self.dw_bn is not None:
            dw_out = dw_out + self.dw_bn(x)      # identity residual (BN only)
        dw_out = self.act(dw_out)

        # ── Pointwise stage ───────────────────────────────────────────────────
        pw_out = sum(branch(dw_out) for branch in self.pw_branches)
        if self.pw_bn is not None:
            pw_out = pw_out + self.pw_bn(dw_out)  # identity residual
        return self.act(pw_out)

    def reparametrize(self) -> None:
        """Fuse all parallel branches into a single conv+bias each.

        After calling this:
          - All branch parameters are deleted (memory freed).
          - self.reparam_dw / self.reparam_pw hold the fused convs.
          - forward() uses the fast inference path.
        """
        if self._is_reparam:
            return

        # ── Fuse depthwise ────────────────────────────────────────────────────
        dw_w = torch.zeros_like(self.dw_branches[0][0].weight)  # [C, 1, 3, 3] for groups=C
        dw_b = torch.zeros(self.in_channels, device=dw_w.device, dtype=dw_w.dtype)

        for branch in self.dw_branches:
            w, b = _fuse_bn(branch[0], branch[1])
            dw_w = dw_w + w
            dw_b = dw_b + b

        if self.dw_skip_1x1 is not None:
            w, b = _fuse_bn(self.dw_skip_1x1[0], self.dw_skip_1x1[1])
            # Pad 1×1 weight to 3×3 (centre) to match branch kernels
            w_padded = F.pad(w, [1, 1, 1, 1])
            dw_w = dw_w + w_padded
            dw_b = dw_b + b

        if self.dw_bn is not None:
            # Identity branch: weight is an identity 3×3 depthwise kernel
            eye_w = _identity_dw_kernel(self.in_channels, device=dw_w.device, dtype=dw_w.dtype)
            scale = self.dw_bn.weight / (self.dw_bn.running_var + self.dw_bn.eps).sqrt()
            bias  = self.dw_bn.bias - self.dw_bn.running_mean * scale
            dw_w  = dw_w + eye_w * scale.view(-1, 1, 1, 1)
            dw_b  = dw_b + bias

        self.reparam_dw = nn.Conv2d(
            self.in_channels, self.in_channels, kernel_size=3,
            stride=self.stride, padding=1, groups=self.in_channels, bias=True,
        )
        self.reparam_dw.weight.data = dw_w
        self.reparam_dw.bias.data   = dw_b

        # ── Fuse pointwise ────────────────────────────────────────────────────
        pw_w = torch.zeros_like(self.pw_branches[0][0].weight)  # [C_out, C_in, 1, 1]
        pw_b = torch.zeros(self.out_channels, device=pw_w.device, dtype=pw_w.dtype)

        for branch in self.pw_branches:
            w, b = _fuse_bn(branch[0], branch[1])
            pw_w = pw_w + w
            pw_b = pw_b + b

        if self.pw_bn is not None and self.in_channels == self.out_channels:
            # Identity branch: weight is 1×1 identity (eye) for pointwise
            eye_w = _identity_pw_kernel(self.in_channels, device=pw_w.device, dtype=pw_w.dtype)
            scale = self.pw_bn.weight / (self.pw_bn.running_var + self.pw_bn.eps).sqrt()
            bias  = self.pw_bn.bias - self.pw_bn.running_mean * scale
            pw_w  = pw_w + eye_w * scale.view(-1, 1, 1, 1)
            pw_b  = pw_b + bias

        self.reparam_pw = nn.Conv2d(
            self.in_channels, self.out_channels, kernel_size=1, bias=True,
        )
        self.reparam_pw.weight.data = pw_w
        self.reparam_pw.bias.data   = pw_b

        # ── Delete training-time branches (free memory) ───────────────────────
        del self.dw_branches, self.dw_skip_1x1, self.dw_bn
        del self.pw_branches, self.pw_bn
        self._is_reparam = True


def _identity_dw_kernel(channels: int, device, dtype) -> torch.Tensor:
    """Create a 3×3 depthwise identity kernel: 1 at centre, 0 elsewhere."""
    w = torch.zeros(channels, 1, 3, 3, device=device, dtype=dtype)
    w[:, 0, 1, 1] = 1.0
    return w


def _identity_pw_kernel(channels: int, device, dtype) -> torch.Tensor:
    """Create a 1×1 pointwise identity kernel: eye matrix reshaped."""
    w = torch.zeros(channels, channels, 1, 1, device=device, dtype=dtype)
    for i in range(channels):
        w[i, i, 0, 0] = 1.0
    return w


# ── MobileOne Backbone ────────────────────────────────────────────────────────

class MobileOneBackbone(BaseVisionBackbone):
    """MobileOne hierarchical CNN backbone.

    4 stages with progressively wider channels and stride-2 downsampling.
    Multi-scale feature maps returned for FPN/BEV/detection heads.

    Args:
        width_mult: Per-stage width multipliers over base channels [48, 96, 192, 512].
        num_blocks: k branches per block (1 = reparam-ready; 4 = rich training).
        img_size, in_chans, drop_rate: See BaseVisionBackbone.
    """

    arch_name:    str       = "mobile_one"
    embed_dim:    int       = 512          # overridden per variant
    feature_dims: list[int] = []

    _BASE_CHANNELS = [48, 96, 192, 512]    # stage output channels before width scaling
    _DEPTHS        = [2, 8, 10, 1]         # number of blocks per stage

    def __init__(
        self,
        width_mult:  list[float] = (1.0, 1.0, 1.0, 1.0),
        num_blocks:  int         = 1,
        img_size:    int         = 224,
        in_chans:    int         = 3,
        drop_rate:   float       = 0.0,
        out_indices: list[int] | None = None,
    ) -> None:
        # Compute actual channel counts
        stage_dims = [
            _make_divisible(c * m, 8)
            for c, m in zip(self._BASE_CHANNELS, width_mult)
        ]
        self.feature_dims = stage_dims
        self.embed_dim    = stage_dims[-1]

        super().__init__(img_size=img_size, in_chans=in_chans,
                         drop_rate=drop_rate, out_indices=out_indices)

        self._num_blocks = num_blocks

        # ── Stem (stride-2 conv + MobileOneBlock) ────────────────────────────
        stem_out = _make_divisible(self._BASE_CHANNELS[0] * width_mult[0] / 2, 8)
        self.stem = nn.Sequential(
            _ConvBN(in_chans, stem_out, kernel_size=3, stride=2),
            nn.ReLU(),
            MobileOneBlock(stem_out, stage_dims[0], stride=2, num_blocks=num_blocks),
        )

        # ── Stages ───────────────────────────────────────────────────────────
        # Stage 0 is produced by stem; stages 1-3 further downsample
        self.stages: nn.ModuleList = nn.ModuleList()
        in_c = stage_dims[0]
        for i, (out_c, n_depth) in enumerate(zip(stage_dims[1:], self._DEPTHS[1:]), start=1):
            blocks: list[nn.Module] = []
            # First block: stride 2 to halve resolution
            blocks.append(MobileOneBlock(in_c, out_c, stride=2, num_blocks=num_blocks))
            # Remaining blocks: same resolution
            for _ in range(n_depth - 1):
                blocks.append(MobileOneBlock(out_c, out_c, stride=1, num_blocks=num_blocks))
            self.stages.append(nn.Sequential(*blocks))
            in_c = out_c

        # ── Global pooling for classification/embedding ───────────────────────
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_drop = nn.Dropout(drop_rate)

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
        # Stage 0 output from stem
        x = self.stem(x)
        all_features: list[torch.Tensor] = [x]

        for stage in self.stages:
            x = stage(x)
            all_features.append(x)

        # Select requested stages
        indices = self.out_indices if self.out_indices is not None else list(range(4))
        feature_maps = [all_features[i] for i in indices if i < len(all_features)]

        # Global pooled embedding
        pooled = self.pool(all_features[-1]).flatten(1)
        pooled = self.head_drop(pooled)

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=None,
            feature_maps=feature_maps,
        )

    def reparametrize(self) -> None:
        """Fuse all multi-branch blocks for inference (run once before deployment)."""
        for m in self.modules():
            if isinstance(m, MobileOneBlock):
                m.reparametrize()

    def freeze_up_to_stage(self, stage: int) -> None:
        """Freeze stem and the first `stage` stages (0-indexed)."""
        for p in self.stem.parameters():
            p.requires_grad_(False)
        for i, s in enumerate(self.stages):
            if i < stage:
                for p in s.parameters():
                    p.requires_grad_(False)


def _make_divisible(v: float, divisor: int) -> int:
    """Round v up to the nearest multiple of divisor (min divisor)."""
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


# ── Variant Registration ──────────────────────────────────────────────────────

@register_vision_backbone("mobile_one_s0")
class MobileOneS0(MobileOneBackbone):
    """MobileOne-S0: smallest variant, 1.6ms on iPhone 12."""
    arch_name = "mobile_one_s0"
    def __init__(self, **kwargs):
        super().__init__(width_mult=[0.75, 1.0, 1.0, 2.0], num_blocks=1, **kwargs)


@register_vision_backbone("mobile_one_s1")
class MobileOneS1(MobileOneBackbone):
    """MobileOne-S1: balanced accuracy/speed."""
    arch_name = "mobile_one_s1"
    def __init__(self, **kwargs):
        super().__init__(width_mult=[1.5, 1.5, 2.0, 2.5], num_blocks=1, **kwargs)


@register_vision_backbone("mobile_one_s2")
class MobileOneS2(MobileOneBackbone):
    """MobileOne-S2: higher accuracy, ~2ms."""
    arch_name = "mobile_one_s2"
    def __init__(self, **kwargs):
        super().__init__(width_mult=[1.5, 2.0, 2.5, 4.0], num_blocks=1, **kwargs)


@register_vision_backbone("mobile_one_s3")
class MobileOneS3(MobileOneBackbone):
    """MobileOne-S3: large efficient model."""
    arch_name = "mobile_one_s3"
    def __init__(self, **kwargs):
        super().__init__(width_mult=[2.0, 2.5, 3.0, 4.0], num_blocks=1, **kwargs)


@register_vision_backbone("mobile_one_s4")
class MobileOneS4(MobileOneBackbone):
    """MobileOne-S4: largest, highest accuracy in the family."""
    arch_name = "mobile_one_s4"
    def __init__(self, **kwargs):
        super().__init__(width_mult=[3.0, 3.5, 3.5, 4.0], num_blocks=1, **kwargs)
