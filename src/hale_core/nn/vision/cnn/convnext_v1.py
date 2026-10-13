"""
ConvNeXt V1 Vision Backbones
=============================

Implements the ConvNeXt V1 family (Liu et al. 2022, "A ConvNet for the 2020s",
CVPR 2022, https://arxiv.org/abs/2201.03545).

Design philosophy
-----------------
ConvNeXt V1 is a pure CNN redesigned to systematically match Swin Transformer.
Starting from ResNet-50, the authors apply a sequence of design improvements:

  1. Patchify stem: 4×4 stride-4 conv (no 7×7 + maxpool).
  2. Stage compute ratio: [3,3,9,3] matching Swin's stage distribution.
  3. Inverted bottleneck: expand channels 4× inside each block (like MobileNet).
  4. Depthwise 7×7 conv: matches Swin's 7×7 local window receptive field.
  5. Fewer activations: only one GELU (after the expansion linear).
  6. Fewer normalisations: only one LayerNorm per block (before the linear layers).
  7. LayerNorm instead of BatchNorm: matches transformer-style normalisation.
  8. Separate downsampling: LayerNorm + 2×2 stride-2 conv between stages.
  9. Layer Scale: learnable per-channel scalar γ (init 1e-6) stabilises training
     of deep networks; enabled for large/xlarge.

Block structure (ConvNeXtV1Block)
----------------------------------
  input [B, C, H, W]  (channels-first)
    → DWConv(C, 7×7, groups=C)
    → permute to [B, H, W, C]
    → LayerNorm(C, ε=1e-6)
    → Linear(C → 4C)
    → GELU
    → Linear(4C → C)
    → * layer_scale_gamma  (if layer_scale_init_value > 0)
    → permute back to [B, C, H, W]
    → stochastic depth + residual

Difference from V2
------------------
  V1: no GRN (Global Response Normalization).
  V2: adds GRN after the wide (4C) GELU layer.
  V1: optional Layer Scale (per-channel scalar).
  V2: replaces Layer Scale with GRN.

Registered names
----------------
  convnext_v1_tiny   — depths=[3,3,9,3],   dim=96
  convnext_v1_small  — depths=[3,3,27,3],  dim=96
  convnext_v1_base   — depths=[3,3,27,3],  dim=128
  convnext_v1_large  — depths=[3,3,27,3],  dim=192  (layer_scale enabled)
  convnext_v1_xlarge — depths=[3,3,27,3],  dim=256  (layer_scale enabled)

Citations
---------
  Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022).
  A ConvNet for the 2020s. CVPR.  arXiv:2201.03545

  Woo, S. et al. (2023). ConvNeXt V2: Co-designing and Scaling ConvNets with
  Masked Autoencoders. CVPR.  arXiv:2301.00808
"""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.registry import register_vision_backbone


# ── LayerNorm for channels-first tensors ──────────────────────────────────────

class LayerNorm2d(nn.LayerNorm):
    """LayerNorm adapted for [B, C, H, W] spatial tensors.

    Transposes to channels-last before normalising, then back to channels-first.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)   # → [B, H, W, C]
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)  # → [B, C, H, W]


# ── Stochastic depth ──────────────────────────────────────────────────────────

class StochasticDepth(nn.Module):
    """Randomly drops the entire block output during training (Huang et al. 2016)."""

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


# ── ConvNeXt V1 block ─────────────────────────────────────────────────────────

class ConvNeXtV1Block(nn.Module):
    """Single ConvNeXt V1 residual block (no GRN, optional Layer Scale).

    Args:
        dim:                    Number of input/output channels.
        drop_path:              Stochastic depth probability.
        layer_scale_init_value: Initial value of per-channel layer scale γ.
                                0 disables layer scale (tiny/small/base).
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 0.0,
    ) -> None:
        super().__init__()

        # Depthwise 7×7 conv: spatial mixing within each channel
        self.dwconv  = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm    = nn.LayerNorm(dim, eps=1e-6)
        # Inverted bottleneck: expand 4× then contract
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act     = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # Layer Scale: per-channel learnable scalar (replaces nothing, just scales)
        self.gamma: nn.Parameter | None = None
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(
                torch.full((dim,), layer_scale_init_value)
            )

        self.drop_path = StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.dwconv(x)                 # [B, C, H, W]
        x = x.permute(0, 2, 3, 1)         # → [B, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)                # [B, H, W, 4C]
        x = self.act(x)
        x = self.pwconv2(x)                # [B, H, W, C]

        if self.gamma is not None:
            x = x * self.gamma             # layer scale

        x = x.permute(0, 3, 1, 2)         # → [B, C, H, W]
        return residual + self.drop_path(x)


# ── ConvNeXt V1 backbone ──────────────────────────────────────────────────────

class ConvNeXtV1(BaseVisionBackbone):
    """Full ConvNeXt V1 feature extractor.

    Args:
        depths:                  Number of blocks in each of the 4 stages.
        dim:                     Base channel width (first stage).
        drop_path_rate:          Max stochastic depth rate (linearly distributed).
        layer_scale_init_value:  Initial layer scale γ (0 = disabled).
        img_size:                Expected input resolution.
        in_chans:                Input channels (default 3).
        drop_rate:               Dropout after global pool.
        out_indices:             Which stage feature maps to return.
    """

    def __init__(
        self,
        depths:                  list[int],
        dim:                     int,
        drop_path_rate:          float = 0.0,
        layer_scale_init_value:  float = 0.0,
        img_size:                int   = 224,
        in_chans:                int   = 3,
        drop_rate:               float = 0.0,
        out_indices:             list[int] | None = None,
    ) -> None:
        super().__init__(img_size=img_size, in_chans=in_chans, drop_rate=drop_rate, out_indices=out_indices)

        dims = [dim, 2 * dim, 4 * dim, 8 * dim]
        self.embed_dim    = dims[-1]
        self.feature_dims = dims

        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # ── Stem: patchify (4×4, stride 4) ───────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6),
        )

        # ── 4 main stages ─────────────────────────────────────────────────────
        self.stages = nn.ModuleList()
        block_idx = 0
        for depth, d in zip(depths, dims):
            blocks = [
                ConvNeXtV1Block(
                    dim=d,
                    drop_path=dp_rates[block_idx + j],
                    layer_scale_init_value=layer_scale_init_value,
                )
                for j in range(depth)
            ]
            self.stages.append(nn.Sequential(*blocks))
            block_idx += depth

        # ── Downsampling between stages ───────────────────────────────────────
        self.downsamples = nn.ModuleList([
            nn.Sequential(
                LayerNorm2d(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            for i in range(len(dims) - 1)
        ])

        # ── Head ─────────────────────────────────────────────────────────────
        self.norm    = nn.LayerNorm(dims[-1], eps=1e-6)
        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """Forward pass: stem → 4 stages with downsampling → pool.

        Returns:
            BackboneOutput with 4 feature maps at strides 4, 8, 16, 32.
        """
        x = self.stem(x)   # [B, dim, H/4, W/4]

        feature_maps = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            feature_maps.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        pooled = self.pool(feature_maps[-1]).flatten(1)  # [B, 8*dim]
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)

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
            if i < len(self.downsamples):
                for p in self.downsamples[i].parameters():
                    p.requires_grad_(False)


# ── Registered variants ───────────────────────────────────────────────────────

@register_vision_backbone("convnext_v1_tiny")
class ConvNeXtV1Tiny(ConvNeXtV1):
    """ConvNeXt V1 Tiny: depths=[3,3,9,3], dim=96 — 28M params, 82.1% top-1."""
    arch_name    = "convnext_v1_tiny"
    embed_dim    = 768
    feature_dims = [96, 192, 384, 768]

    def __init__(self, **kwargs) -> None:
        super().__init__(depths=[3, 3, 9, 3], dim=96,
                         layer_scale_init_value=0.0, **kwargs)


@register_vision_backbone("convnext_v1_small")
class ConvNeXtV1Small(ConvNeXtV1):
    """ConvNeXt V1 Small: depths=[3,3,27,3], dim=96 — 50M params, 83.1% top-1."""
    arch_name    = "convnext_v1_small"
    embed_dim    = 768
    feature_dims = [96, 192, 384, 768]

    def __init__(self, **kwargs) -> None:
        super().__init__(depths=[3, 3, 27, 3], dim=96,
                         layer_scale_init_value=0.0, **kwargs)


@register_vision_backbone("convnext_v1_base")
class ConvNeXtV1Base(ConvNeXtV1):
    """ConvNeXt V1 Base: depths=[3,3,27,3], dim=128 — 89M params, 83.8% top-1."""
    arch_name    = "convnext_v1_base"
    embed_dim    = 1024
    feature_dims = [128, 256, 512, 1024]

    def __init__(self, **kwargs) -> None:
        super().__init__(depths=[3, 3, 27, 3], dim=128,
                         layer_scale_init_value=0.0, **kwargs)


@register_vision_backbone("convnext_v1_large")
class ConvNeXtV1Large(ConvNeXtV1):
    """ConvNeXt V1 Large: depths=[3,3,27,3], dim=192, layer_scale=1e-6 — 198M params, 84.3%."""
    arch_name    = "convnext_v1_large"
    embed_dim    = 1536
    feature_dims = [192, 384, 768, 1536]

    def __init__(self, **kwargs) -> None:
        super().__init__(depths=[3, 3, 27, 3], dim=192,
                         layer_scale_init_value=1e-6, **kwargs)


@register_vision_backbone("convnext_v1_xlarge")
class ConvNeXtV1XLarge(ConvNeXtV1):
    """ConvNeXt V1 XLarge: depths=[3,3,27,3], dim=256, layer_scale=1e-6 — 350M params, 87.0%."""
    arch_name    = "convnext_v1_xlarge"
    embed_dim    = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__(depths=[3, 3, 27, 3], dim=256,
                         layer_scale_init_value=1e-6, **kwargs)
