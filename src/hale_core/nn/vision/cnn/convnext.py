"""
ConvNeXt V2 Vision Backbones
=============================

Implements the ConvNeXt V2 family (Woo et al. 2023, "ConvNeXt V2: Co-designing and
Scaling ConvNets with Masked Autoencoders").

Architecture evolution
----------------------
ConvNeXt V1 (Liu et al. 2022, "A ConvNet for the 2020s"):
  - Modernised ConvNet: depthwise 7×7, inverted bottleneck, GeLU, LayerNorm.
  - Matched ViT in accuracy and design style while remaining a pure CNN.

ConvNeXt V2 adds:
  - FCMAE (Fully Convolutional Masked Autoencoder) pre-training paradigm.
  - Global Response Normalization (GRN): a channel feature competition mechanism
    analogous to SE (Squeeze-Excite) but operating globally on each channel.

ConvNeXtV2Block
---------------
  Input [B, C, H, W]
  1. Depthwise 7×7 conv (groups=C) — captures long-range spatial structure.
     Channels-first → channels-last transpose for LayerNorm efficiency.
  2. LayerNorm (channels-last, ε=1e-6).
  3. Pointwise Linear expand: C → 4C.
  4. GeLU activation.
  5. Global Response Normalization (GRN):
       x_norm = ‖x‖_2 across spatial dims → [B, 1, 1, C]
       x = x * (x_norm / mean(x_norm)) + bias   (gated by spatial norms)
     This penalises redundant channels and encourages diverse feature activations.
  6. Pointwise Linear contract: 4C → C.
  Residual add.

Stage downsampling
------------------
  Stem: 4×4 conv stride 4 → LayerNorm  (initial patch-like tokenization)
  Downsample between stages: LayerNorm + 2×2 conv stride 2.

Registered names and configs
-----------------------------
  convnext_tiny   — depths=[3,3,9,3],    dims=[96,192,384,768]
  convnext_small  — depths=[3,3,27,3],   dims=[96,192,384,768]
  convnext_base   — depths=[3,3,27,3],   dims=[128,256,512,1024]
  convnext_large  — depths=[3,3,27,3],   dims=[192,384,768,1536]
  convnext_xlarge — depths=[3,3,27,3],   dims=[256,512,1024,2048]
  convnext_huge   — depths=[3,3,27,3],   dims=[352,704,1408,2816]

Citations
---------
  Liu et al. (2022). A ConvNet for the 2020s. CVPR.
  Woo et al. (2023). ConvNeXt V2: Co-designing and Scaling ConvNets with MAE. CVPR.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.registry import register_vision_backbone


# ── Layer norm for channels-last tensors ──────────────────────────────────────

class LayerNorm2d(nn.LayerNorm):
    """LayerNorm adapted for channels-first (B, C, H, W) spatial tensors.

    PyTorch's nn.LayerNorm expects the normalised dims to be the last dims.
    This wrapper transposes to channels-last before normalising, then transposes back.
    This is the standard approach used by ConvNeXt and many modern CNNs.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = x.permute(0, 2, 3, 1)          # → [B, H, W, C]
        x = super().forward(x)              # LayerNorm over last dim C
        return x.permute(0, 3, 1, 2)       # → [B, C, H, W]


# ── Global Response Normalization ────────────────────────────────────────────

class GRN(nn.Module):
    """Global Response Normalization (GRN) — the key addition in ConvNeXt V2.

    GRN introduces cross-channel feature competition: channels with large global
    norms suppress channels with small global norms.  This discourages feature
    collapse (redundant channels learning the same thing) during masked autoencoding.

    Formulation (Woo et al. 2023, Eq. 1–3):
        gx = ‖x‖₂ over H, W dims          → [B, 1, 1, C]  (global spatial norm)
        nx = gx / E[gx]                    → normalised norms (E over C)
        output = x * nx + gamma * x + beta
    where gamma and beta are learned per-channel parameters (shape [C]).

    Args:
        dim: Number of feature channels (C).

    Shape:
        Input / Output: [B, H, W, C]  (channels-last convention inside blocks)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        # Learnable scale and bias initialised to 0 (start as identity)
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta  = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global spatial L2 norm per channel: [B, 1, 1, C]
        gx = x.norm(p=2, dim=(1, 2), keepdim=True)
        # Normalise across the channel dimension
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        # Scale and shift
        return self.gamma * (x * nx) + self.beta + x


# ── ConvNeXt V2 block ─────────────────────────────────────────────────────────

class ConvNeXtV2Block(nn.Module):
    """Single ConvNeXt V2 residual block.

    Sequence: DWConv → LayerNorm → Linear(4x expand) → GeLU → GRN → Linear(contract)
    with a residual connection.  All linear ops are implemented as pointwise 1×1
    convolutions via nn.Linear on the transposed tensor (channels-last) for clarity.

    Args:
        dim:        Number of input (and output) channels.
        drop_path:  Stochastic depth drop probability (0 = disabled).
    """

    def __init__(self, dim: int, drop_path: float = 0.0) -> None:
        super().__init__()

        # Depthwise 7×7 conv — processes spatial context within each channel
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # LayerNorm in channels-last space (after transpose)
        self.norm  = nn.LayerNorm(dim, eps=1e-6)
        # Inverted bottleneck: expand 4× for more capacity
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act     = nn.GELU()
        # GRN — the V2-specific feature competition layer
        self.grn     = GRN(4 * dim)
        # Contract back to dim
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # Stochastic depth: randomly drops the entire block during training
        # (equivalent to randomly skipping the residual branch)
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.dwconv(x)                  # [B, C, H, W] — depthwise spatial conv
        x = x.permute(0, 2, 3, 1)          # → [B, H, W, C] for LayerNorm + Linear
        x = self.norm(x)
        x = self.pwconv1(x)                 # [B, H, W, 4C]
        x = self.act(x)
        x = self.grn(x)                     # feature competition
        x = self.pwconv2(x)                 # [B, H, W, C]
        x = x.permute(0, 3, 1, 2)          # → [B, C, H, W]

        return residual + self.drop_path(x)


# ── Stochastic depth (drop path) ─────────────────────────────────────────────

class StochasticDepth(nn.Module):
    """Stochastic depth regularisation (Huang et al. 2016).

    Randomly drops the entire forward path of a block during training,
    equivalent to randomly shortening the effective depth of the network.
    At inference, acts as identity scaled by survival probability.

    Args:
        drop_prob: Probability of dropping (zeroing) the block output.
    """

    def __init__(self, drop_prob: float) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Random binary mask: shape [B, 1, 1, 1] — drops entire samples in the batch
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.rand(shape, dtype=x.dtype, device=x.device)
        noise = torch.floor(noise + keep_prob)
        # Scale by 1/keep_prob to maintain expected value (like dropout)
        return x * noise / keep_prob


# ── ConvNeXt V2 backbone ──────────────────────────────────────────────────────

class ConvNeXtV2(BaseVisionBackbone):
    """Full ConvNeXt V2 feature extractor.

    Builds 4 stages of ConvNeXtV2Block with progressive downsampling,
    returning 4-scale feature maps for FPN/detection heads and a global
    pooled embedding for classification/VLM heads.

    Args:
        depths:      Number of blocks in each of the 4 stages.
        dims:        Channel dimensions at each of the 4 stages.
        drop_path_rate: Maximum stochastic depth rate (linearly distributed).
        img_size:    Expected input spatial resolution.
        in_chans:    Input image channels (default 3).
        drop_rate:   Dropout after global pool.
        out_indices: Which stage outputs to include in feature_maps.
    """

    def __init__(
        self,
        depths:          list[int],
        dims:            list[int],
        drop_path_rate:  float = 0.0,
        img_size:        int   = 224,
        in_chans:        int   = 3,
        drop_rate:       float = 0.0,
        out_indices:     list[int] | None = None,
    ) -> None:
        super().__init__(img_size=img_size, in_chans=in_chans, drop_rate=drop_rate, out_indices=out_indices)

        self.embed_dim   = dims[-1]
        self.feature_dims = list(dims)

        # Linearly distribute drop_path rates across all blocks
        total_blocks = sum(depths)
        dp_rates = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, total_blocks)
        ]

        # ── Stem: patch-like tokenisation (4×4 stride-4) ─────────────────────
        # Unlike ViT's patch embedding, this is a simple large-stride conv + LN.
        # Reduces [B, 3, H, W] → [B, dims[0], H/4, W/4]
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6),
        )

        # ── 4 main stages ─────────────────────────────────────────────────────
        self.stages = nn.ModuleList()
        block_idx = 0
        for i, (depth, dim) in enumerate(zip(depths, dims)):
            blocks = [
                ConvNeXtV2Block(dim, drop_path=dp_rates[block_idx + j])
                for j in range(depth)
            ]
            self.stages.append(nn.Sequential(*blocks))
            block_idx += depth

        # ── Downsampling between stages (stages 0→1, 1→2, 2→3) ───────────────
        # LayerNorm + 2×2 stride-2 conv halves resolution and doubles channels
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
        """Truncated normal for conv/linear; const for norm layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """Forward pass returning multi-scale feature maps and pooled embedding.

        Args:
            x: [B, in_chans, H, W]

        Returns:
            BackboneOutput:
              pooled_output — [B, dims[-1]]
              patch_tokens  — None
              feature_maps  — 4 tensors at strides 4, 8, 16, 32
        """
        x = self.stem(x)   # [B, dims[0], H/4, W/4]

        feature_maps = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            feature_maps.append(x)          # collect output of each stage
            if i < len(self.downsamples):   # downsample between stages (not after last)
                x = self.downsamples[i](x)

        # Global pooled embedding
        pooled = self.pool(feature_maps[-1]).flatten(1)  # [B, dims[-1]]
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)

        # Select requested scales
        all_maps = feature_maps
        if self.out_indices is not None:
            all_maps = [feature_maps[i] for i in self.out_indices]

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=None,
            feature_maps=all_maps,
            attentions=None,
        )

    def freeze_up_to_stage(self, stage: int) -> None:
        """Freeze stem and stages 0 through `stage` inclusive.

        Args:
            stage: 0-based index in [0, 3].
        """
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

@register_vision_backbone("convnext_tiny")
class ConvNeXtTiny(ConvNeXtV2):
    """ConvNeXt V2 Tiny: depths=[3,3,9,3], dims=[96,192,384,768]."""
    arch_name    = "convnext_tiny"
    embed_dim    = 768
    feature_dims = [96, 192, 384, 768]

    def __init__(self, **kwargs) -> None:
        super().__init__(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)


@register_vision_backbone("convnext_small")
class ConvNeXtSmall(ConvNeXtV2):
    """ConvNeXt V2 Small: depths=[3,3,27,3], dims=[96,192,384,768]."""
    arch_name    = "convnext_small"
    embed_dim    = 768
    feature_dims = [96, 192, 384, 768]

    def __init__(self, **kwargs) -> None:
        super().__init__(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)


@register_vision_backbone("convnext_base")
class ConvNeXtBase(ConvNeXtV2):
    """ConvNeXt V2 Base: depths=[3,3,27,3], dims=[128,256,512,1024]."""
    arch_name    = "convnext_base"
    embed_dim    = 1024
    feature_dims = [128, 256, 512, 1024]

    def __init__(self, **kwargs) -> None:
        super().__init__(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)


@register_vision_backbone("convnext_large")
class ConvNeXtLarge(ConvNeXtV2):
    """ConvNeXt V2 Large: depths=[3,3,27,3], dims=[192,384,768,1536]."""
    arch_name    = "convnext_large"
    embed_dim    = 1536
    feature_dims = [192, 384, 768, 1536]

    def __init__(self, **kwargs) -> None:
        super().__init__(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)


@register_vision_backbone("convnext_xlarge")
class ConvNeXtXLarge(ConvNeXtV2):
    """ConvNeXt V2 XLarge: depths=[3,3,27,3], dims=[256,512,1024,2048]."""
    arch_name    = "convnext_xlarge"
    embed_dim    = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)


@register_vision_backbone("convnext_huge")
class ConvNeXtHuge(ConvNeXtV2):
    """ConvNeXt V2 Huge: depths=[3,3,27,3], dims=[352,704,1408,2816]."""
    arch_name    = "convnext_huge"
    embed_dim    = 2816
    feature_dims = [352, 704, 1408, 2816]

    def __init__(self, **kwargs) -> None:
        super().__init__(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
