"""
InternImage — DCNv3-based Vision Backbone (Wang et al. 2022)
=============================================================

Reference: "InternImage: Exploring Large-Scale Vision Foundation Models
           with Deformable Convolutions"
           https://arxiv.org/abs/2211.05778

Key Innovation: DCNv3 (Deformable Convolution v3)
--------------------------------------------------
DCNv3 makes deformable convolution efficient and scalable by redesigning it
as a SEPARABLE operation, analogous to the ViT attention mechanism:

Standard DCNv2 (each of G groups has its own fully-connected projection):
  - Output = Σ_g Σ_k w_g,k · x(p + Δp_g,k) · m_g,k
  - Projection per group: O(C²) cost

DCNv3 (shared depthwise projection, separate per-group offsets/masks):
  - Depthwise over all channels first (shared, O(C) cost)
  - Then per-group offset/mask prediction (O(C) total, not O(C²))
  - Output = Σ_g Σ_k (W_proj · x(p + Δp_g,k)) · m_g,k
  - This makes DCNv3 as scalable as ViT attention

Deformable Convolution Sampling:
  Instead of a fixed k×k grid, DCNv3 learns offsets Δp_g,k for each
  of K sampling points in each of G groups. The network learns WHERE
  to look in the input feature map for each output location.
  - Input position p: any spatial location in the feature map
  - Learned offset Δp_g,k: 2D offset for sampling point k in group g
  - Sampling: x(p + Δp_g,k) via bilinear interpolation
  - Modulation mask m_g,k: scalar in [0,1] weighting each sample point

  This gives adaptive receptive fields: for cars, the network might learn
  to aggregate features from the wheels and roof; for text, each character
  might focus on adjacent strokes.

DCNv3 Layer Implementation (without the C extension):
  We implement DCNv3 using F.grid_sample (bilinear interpolation) on
  GPU-computed offset coordinates. This is slightly slower than the CUDA
  kernel but avoids the build dependency on DCNv3's custom ops.

Architecture
------------
  Stem: 4×4 stride-4 conv (patch-like), LayerNorm
  4 stages, each with:
    - N InternImage blocks (DCNv3 + FFN)
    - Downsampling layer between stages (2×2 stride-2 conv, LayerNorm)

  InternImageBlock:
    input → LayerNorm → DCNv3 → LayerNorm → FFN → residual add

  FFN: Linear → GELU → Linear (4× expand)

Registered variants:
  internimage_tiny:  C=64,  depths=[4,4,18,4],  groups=[4,8,16,32]
  internimage_small: C=80,  depths=[4,4,21,4],  groups=[5,10,20,40]
  internimage_base:  C=112, depths=[4,4,21,4],  groups=[7,14,28,56]
  internimage_large: C=160, depths=[5,5,22,5],  groups=[10,20,40,80]
  internimage_huge:  C=320, depths=[6,6,32,6],  groups=[20,40,80,160]
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.registry import register_vision_backbone


# ── DCNv3 Core Implementation ─────────────────────────────────────────────────

class DCNv3(nn.Module):
    """Deformable Convolution v3 — separable formulation.

    Implements DCNv3 using torch.nn.functional.grid_sample as the
    bilinear interpolation backend. This avoids the custom CUDA kernel
    dependency while preserving the mathematical equivalence.

    For production deployments with the original InternImage CUDA kernel,
    replace the sample_with_offsets() call below with the custom op.

    Args:
        channels:    Feature channel count (total, split across groups).
        kernel_size: Sampling grid side length k; total K = k² sampling points.
        stride:      Spatial stride (1 = same-size, 2 = downsample).
        groups:      Number of deformable groups (must divide channels evenly).
        offset_scale: Multiplier for learned offsets (controls receptive field).
        dw_kernel:   Kernel size for the depthwise pre-projection.
    """

    def __init__(
        self,
        channels:     int,
        kernel_size:  int   = 3,
        stride:       int   = 1,
        groups:       int   = 4,
        offset_scale: float = 1.0,
        dw_kernel:    int   = 3,
    ) -> None:
        super().__init__()
        assert channels % groups == 0, "channels must be divisible by groups"

        self.channels     = channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.groups       = groups
        self.offset_scale = offset_scale
        self.K            = kernel_size * kernel_size  # total sampling points

        # Depthwise pre-projection (shared across groups, O(C) cost)
        self.dw_conv = nn.Sequential(
            nn.Conv2d(channels, channels, dw_kernel,
                      stride=1, padding=dw_kernel // 2, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        # Per-group offset + mask prediction:
        # For each output location: G groups × K points × 2 offsets + G×K masks
        self.offset_mask_proj = nn.Linear(channels, groups * self.K * 3)
        # Bias init: zero offsets, positive mask logits → mild deformation at start
        nn.init.zeros_(self.offset_mask_proj.weight)
        nn.init.constant_(self.offset_mask_proj.bias, 0.0)

        # Output projection: per-group → full channels
        self.output_proj = nn.Linear(channels, channels)

        self._init_base_grid()

    def _init_base_grid(self) -> None:
        """Pre-compute the regular k×k grid of sampling offsets (centre = 0).

        Base grid: for a 3×3 kernel → 9 points at offsets (-1,-1), (-1,0), ...
        These are added to the learned Δp to get final sampling positions.
        """
        ks = self.kernel_size
        # Base offsets: [K, 2] in (y, x) order
        ys = torch.arange(ks, dtype=torch.float32) - ks // 2
        xs = torch.arange(ks, dtype=torch.float32) - ks // 2
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        base_grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2)  # [K, 2]
        self.register_buffer("base_grid", base_grid, persistent=False)  # [K, 2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply DCNv3 deformable convolution.

        Args:
            x: [B, C, H, W] input features.

        Returns:
            [B, C, H_out, W_out] output features.
            H_out = H/stride, W_out = W/stride.
        """
        B, C, H, W = x.shape
        Hs = math.ceil(H / self.stride)
        Ws = math.ceil(W / self.stride)

        # ── Step 1: Depthwise pre-projection ────────────────────────────────
        # Shared feature aggregation before per-group deformable sampling
        x_dw = self.dw_conv(x)  # [B, C, H, W]

        # Pool to output resolution if stride > 1
        if self.stride > 1:
            x_pool = F.avg_pool2d(x_dw, kernel_size=self.stride, stride=self.stride)
        else:
            x_pool = x_dw  # [B, C, Hs, Ws]

        # ── Step 2: Predict offsets and modulation masks ─────────────────────
        # Reshape spatial dims to token sequence for linear proj
        x_flat = x_pool.permute(0, 2, 3, 1).reshape(B * Hs * Ws, C)
        offset_mask = self.offset_mask_proj(x_flat)           # [B*Hs*Ws, G*K*3]
        offset_mask = offset_mask.view(B, Hs, Ws, self.groups, self.K, 3)

        # Split into offsets (2D) and masks (1D)
        offsets = offset_mask[..., :2] * self.offset_scale    # [B, Hs, Ws, G, K, 2]
        masks   = torch.sigmoid(offset_mask[..., 2])          # [B, Hs, Ws, G, K]

        # ── Step 3: Bilinear sampling at offset positions ────────────────────
        x_sampled = self._deformable_sample(x, offsets, Hs, Ws)
        # x_sampled: [B, G, K, Hs, Ws, C//G]

        # ── Step 4: Apply modulation masks and aggregate ─────────────────────
        # masks: [B, Hs, Ws, G, K] → [B, G, K, Hs, Ws, 1]
        masks  = masks.permute(0, 3, 4, 1, 2).unsqueeze(-1)
        out    = (x_sampled * masks).sum(dim=2)                # [B, G, Hs, Ws, C//G]
        out    = out.permute(0, 2, 3, 1, 4).reshape(B, Hs, Ws, C)

        # ── Step 5: Output projection ─────────────────────────────────────────
        out = self.output_proj(out)                            # [B, Hs, Ws, C]
        return out.permute(0, 3, 1, 2)                        # [B, C, Hs, Ws]

    def _deformable_sample(
        self,
        x:       torch.Tensor,    # [B, C, H, W]
        offsets: torch.Tensor,    # [B, Hs, Ws, G, K, 2]
        Hs:      int,
        Ws:      int,
    ) -> torch.Tensor:
        """Sample x at offset positions using bilinear interpolation.

        Returns:
            [B, G, K, Hs, Ws, C//G]
        """
        B, C, H, W = x.shape
        G  = self.groups
        K  = self.K
        Cg = C // G

        # Compute sampling grid in normalised [-1, 1] coordinates
        # Base grid: [K, 2] regular k×k grid → broadcast over [B, Hs, Ws, K]
        # offsets:   [B, Hs, Ws, G, K, 2]

        # Create regular grid for output positions.
        # Shape must broadcast with offsets [B, Hs, Ws, G, K, 2], so we add
        # both a G-dim and a K-dim: [1, Hs, Ws, 1, 1, 2].
        ys = torch.linspace(-1 + 1 / Hs, 1 - 1 / Hs, Hs, device=x.device)
        xs = torch.linspace(-1 + 1 / Ws, 1 - 1 / Ws, Ws, device=x.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        grid_base = torch.stack([grid_x, grid_y], dim=-1)      # [Hs, Ws, 2]
        grid_base = grid_base.unsqueeze(0).unsqueeze(3).unsqueeze(4)  # [1, Hs, Ws, 1, 1, 2]

        # Base kernel offsets in normalised coordinates [1, 1, 1, 1, K, 2]
        base_k = self.base_grid.to(x.device)                   # [K, 2]
        base_k_norm = base_k.clone()
        base_k_norm[:, 0] /= (W / 2)
        base_k_norm[:, 1] /= (H / 2)
        base_k_norm = base_k_norm.view(1, 1, 1, 1, K, 2)      # [1, 1, 1, 1, K, 2]

        # Learned offsets (normalised by H/W)
        off_norm = offsets.clone()
        off_norm[..., 0] /= (W / 2)
        off_norm[..., 1] /= (H / 2)

        # Per-group sampling coordinates: broadcast [1,Hs,Ws,1,1,2] + [1,1,1,1,K,2]
        # + [B,Hs,Ws,G,K,2] → [B, Hs, Ws, G, K, 2]
        coords = grid_base + base_k_norm + off_norm
        coords = coords.clamp(-1, 1)

        # Sample x for each group separately.
        # Vectorise over K by treating B*K as the batch dimension for grid_sample.
        x_groups = x.reshape(B, G, Cg, H, W)               # [B, G, Cg, H, W]

        sampled = []
        for g in range(G):
            # Expand input along the K axis: [B, K, Cg, H, W] → [B*K, Cg, H, W]
            xg = x_groups[:, g]                             # [B, Cg, H, W]
            xg = xg.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(B * K, Cg, H, W)

            # Sampling grid for group g: [B, Hs, Ws, K, 2]
            # .contiguous() is needed before reshape because indexing on dim 3 (G)
            # produces a non-contiguous slice.
            cg = coords[:, :, :, g, :, :].contiguous()     # [B, Hs, Ws, K, 2]
            cg = cg.permute(0, 3, 1, 2, 4).reshape(B * K, Hs, Ws, 2)  # [B*K, Hs, Ws, 2]

            out_g = F.grid_sample(xg, cg, mode="bilinear",
                                  align_corners=False, padding_mode="zeros")
            # out_g: [B*K, Cg, Hs, Ws] → [B, K, Cg, Hs, Ws] → [B, Cg, Hs, Ws, K]
            out_g = out_g.reshape(B, K, Cg, Hs, Ws)
            out_g = out_g.permute(0, 2, 3, 4, 1)           # [B, Cg, Hs, Ws, K]
            sampled.append(out_g)

        # Stack groups: [B, G, Cg, Hs, Ws, K]
        out = torch.stack(sampled, dim=1)
        # Rearrange to [B, G, K, Hs, Ws, Cg]
        out = out.permute(0, 1, 5, 3, 4, 2)
        return out


# ── InternImage Building Blocks ───────────────────────────────────────────────

class InternImageFFN(nn.Module):
    """Feed-Forward Network: Linear → GELU → Dropout → Linear → Dropout."""

    def __init__(self, channels: int, expand: int = 4, drop: float = 0.0) -> None:
        super().__init__()
        mid = channels * expand
        self.fc1  = nn.Linear(channels, mid)
        self.act  = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2  = nn.Linear(mid, channels)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class InternImageBlock(nn.Module):
    """Single InternImage transformer block.

    Structure (pre-norm convention):
        x → LN → DCNv3 → x (residual)
        x → LN → FFN   → x (residual)

    Pre-norm stabilises training for large models; same design as ViT.

    Args:
        channels:    Feature channels.
        groups:      DCNv3 deformable groups.
        kernel_size: DCNv3 sampling grid size.
        ffn_expand:  FFN expansion ratio.
        drop:        Dropout for FFN.
        drop_path:   Stochastic depth probability.
    """

    def __init__(
        self,
        channels:    int,
        groups:      int,
        kernel_size: int   = 3,
        ffn_expand:  int   = 4,
        drop:        float = 0.0,
        drop_path:   float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.dcnv3 = DCNv3(channels, kernel_size=kernel_size, groups=groups)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.ffn   = InternImageFFN(channels, ffn_expand, drop)
        # Stochastic depth on both branches
        self.drop_path = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] channels-first
        B, C, H, W = x.shape

        # DCNv3 residual branch (with pre-norm)
        x_ln = x.permute(0, 2, 3, 1)      # → [B, H, W, C] for LayerNorm
        x_ln = self.norm1(x_ln).permute(0, 3, 1, 2)  # → [B, C, H, W]
        x = x + self.drop_path(self.dcnv3(x_ln))

        # FFN residual branch (with pre-norm)
        x_ln = x.permute(0, 2, 3, 1)      # → [B, H, W, C]
        x_ln = self.drop_path(self.ffn(self.norm2(x_ln)))
        x = x + x_ln.permute(0, 3, 1, 2)
        return x


class _DropPath(nn.Module):
    """Stochastic depth: drop entire residual path for a sample."""
    def __init__(self, drop_prob: float) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        survival = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(survival)
        return x * noise.div_(survival)


class InternImageDownsample(nn.Module):
    """Between-stage downsampling: LayerNorm → 3×3 stride-2 conv → double channels."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_ch, eps=1e-6)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = x.permute(0, 2, 3, 1)              # → [B, H, W, C]
        x = self.norm(x).permute(0, 3, 1, 2)   # → [B, C, H, W]
        return self.conv(x)


# ── InternImage Backbone ──────────────────────────────────────────────────────

class InternImageBackbone(BaseVisionBackbone):
    """InternImage hierarchical backbone using DCNv3 operators.

    4 stages, each producing a feature map at a different resolution.
    Adaptive receptive fields via deformable convolution.

    Args:
        stem_channels: Initial channel count (C in the paper).
        depths:        Number of InternImageBlock per stage.
        groups:        DCNv3 deformable groups per stage.
        drop_path_rate: Max stochastic depth rate.
        img_size, in_chans, drop_rate: See BaseVisionBackbone.
    """

    arch_name:    str       = "internimage"
    embed_dim:    int       = 0
    feature_dims: list[int] = []

    def __init__(
        self,
        stem_channels:  int,
        depths:         list[int],
        groups:         list[int],
        drop_path_rate: float        = 0.1,
        img_size:       int          = 224,
        in_chans:       int          = 3,
        drop_rate:      float        = 0.0,
        out_indices:    list[int] | None = None,
    ) -> None:
        assert len(depths) == len(groups) == 4, "Need exactly 4 stages"

        # Channel progression: double each stage
        channels = [stem_channels * (2 ** i) for i in range(4)]
        self.feature_dims = channels
        self.embed_dim    = channels[-1]

        super().__init__(img_size=img_size, in_chans=in_chans,
                         drop_rate=drop_rate, out_indices=out_indices)

        # ── Stem: patch-like embedding with 4×4 stride-4 conv ─────────────────
        # Conv and LayerNorm are separate modules because the LN runs channels-last
        # (see forward). There is deliberately no other stem module: unused parameters
        # never get a gradient and make DistributedDataParallel fail.
        self.stem_conv = nn.Conv2d(in_chans, channels[0], kernel_size=4, stride=4, bias=False)
        self.stem_norm = nn.LayerNorm(channels[0], eps=1e-6)

        # ── Stages ───────────────────────────────────────────────────────────
        total_blocks = sum(depths)
        block_idx    = 0

        self.stages:      nn.ModuleList = nn.ModuleList()
        self.downsamples: nn.ModuleList = nn.ModuleList()

        for i, (n, g) in enumerate(zip(depths, groups)):
            blocks = nn.ModuleList()
            for _ in range(n):
                dp = drop_path_rate * block_idx / max(total_blocks - 1, 1)
                blocks.append(InternImageBlock(
                    channels=channels[i], groups=g,
                    kernel_size=3, ffn_expand=4, drop=0.0, drop_path=dp,
                ))
                block_idx += 1
            self.stages.append(blocks)
            # Downsample between stages (not after last stage)
            if i < 3:
                self.downsamples.append(InternImageDownsample(channels[i], channels[i + 1]))

        # ── Global pooling ────────────────────────────────────────────────────
        self.norm_last = nn.LayerNorm(channels[-1], eps=1e-6)
        self.pool      = nn.AdaptiveAvgPool2d(1)
        self.head_drop = nn.Dropout(drop_rate)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        # Stem
        x = self.stem_conv(x)                                  # [B, C, H/4, W/4]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)                             # channels-last for LN
        x = self.stem_norm(x).permute(0, 3, 1, 2)            # back to [B, C, H, W]

        feature_maps: list[torch.Tensor] = []

        for i, stage_blocks in enumerate(self.stages):
            for block in stage_blocks:
                x = block(x)
            feature_maps.append(x)

            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        # Normalise final features for pooling
        B, C, H, W = feature_maps[-1].shape
        x_last = feature_maps[-1].permute(0, 2, 3, 1)
        x_last = self.norm_last(x_last).permute(0, 3, 1, 2)

        # Global pooled embedding
        pooled = self.pool(x_last).flatten(1)
        pooled = self.head_drop(pooled)

        # Select requested scales
        indices     = self.out_indices if self.out_indices is not None else list(range(4))
        selected    = [feature_maps[i] for i in indices if i < len(feature_maps)]

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=None,
            feature_maps=selected,
        )

    def freeze_up_to_stage(self, stage: int) -> None:
        """Freeze stem and stages 0..stage."""
        for p in self.stem_conv.parameters():
            p.requires_grad_(False)
        for p in self.stem_norm.parameters():
            p.requires_grad_(False)
        for i, s in enumerate(self.stages):
            if i <= stage:
                for p in s.parameters():
                    p.requires_grad_(False)


# ── Registration ─────────────────────────────────────────────────────────────

@register_vision_backbone("internimage_tiny")
class InternImageTiny(InternImageBackbone):
    """InternImage-T: 30M params, fast."""
    arch_name = "internimage_tiny"
    def __init__(self, **kwargs):
        super().__init__(stem_channels=64, depths=[4,4,18,4],
                         groups=[4,8,16,32], **kwargs)


@register_vision_backbone("internimage_small")
class InternImageSmall(InternImageBackbone):
    """InternImage-S: 50M params."""
    arch_name = "internimage_small"
    def __init__(self, **kwargs):
        super().__init__(stem_channels=80, depths=[4,4,21,4],
                         groups=[5,10,20,40], **kwargs)


@register_vision_backbone("internimage_base")
class InternImageBase(InternImageBackbone):
    """InternImage-B: 97M params."""
    arch_name = "internimage_base"
    def __init__(self, **kwargs):
        super().__init__(stem_channels=112, depths=[4,4,21,4],
                         groups=[7,14,28,56], **kwargs)


@register_vision_backbone("internimage_large")
class InternImageLarge(InternImageBackbone):
    """InternImage-L: 223M params, high accuracy."""
    arch_name = "internimage_large"
    def __init__(self, **kwargs):
        super().__init__(stem_channels=160, depths=[5,5,22,5],
                         groups=[10,20,40,80], **kwargs)


@register_vision_backbone("internimage_huge")
class InternImageHuge(InternImageBackbone):
    """InternImage-H: 1.08B params — largest vision backbone in the family."""
    arch_name = "internimage_huge"
    def __init__(self, **kwargs):
        super().__init__(stem_channels=320, depths=[6,6,32,6],
                         groups=[20,40,80,160], **kwargs)
