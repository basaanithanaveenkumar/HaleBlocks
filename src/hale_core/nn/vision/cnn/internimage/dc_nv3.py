"""``DCNv3``: Deformable Convolution v3 — separable formulation."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 4,
        offset_scale: float = 1.0,
        dw_kernel: int = 3,
    ) -> None:
        super().__init__()
        assert channels % groups == 0, "channels must be divisible by groups"

        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.offset_scale = offset_scale
        self.K = kernel_size * kernel_size  # total sampling points

        # Depthwise pre-projection (shared across groups, O(C) cost)
        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                channels,
                channels,
                dw_kernel,
                stride=1,
                padding=dw_kernel // 2,
                groups=channels,
                bias=False,
            ),
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
        offset_mask = self.offset_mask_proj(x_flat)  # [B*Hs*Ws, G*K*3]
        offset_mask = offset_mask.view(B, Hs, Ws, self.groups, self.K, 3)

        # Split into offsets (2D) and masks (1D)
        offsets = offset_mask[..., :2] * self.offset_scale  # [B, Hs, Ws, G, K, 2]
        masks = torch.sigmoid(offset_mask[..., 2])  # [B, Hs, Ws, G, K]

        # ── Step 3: Bilinear sampling at offset positions ────────────────────
        x_sampled = self._deformable_sample(x, offsets, Hs, Ws)
        # x_sampled: [B, G, K, Hs, Ws, C//G]

        # ── Step 4: Apply modulation masks and aggregate ─────────────────────
        # masks: [B, Hs, Ws, G, K] → [B, G, K, Hs, Ws, 1]
        masks = masks.permute(0, 3, 4, 1, 2).unsqueeze(-1)
        out = (x_sampled * masks).sum(dim=2)  # [B, G, Hs, Ws, C//G]
        out = out.permute(0, 2, 3, 1, 4).reshape(B, Hs, Ws, C)

        # ── Step 5: Output projection ─────────────────────────────────────────
        out = self.output_proj(out)  # [B, Hs, Ws, C]
        return out.permute(0, 3, 1, 2)  # [B, C, Hs, Ws]

    def _deformable_sample(
        self,
        x: torch.Tensor,  # [B, C, H, W]
        offsets: torch.Tensor,  # [B, Hs, Ws, G, K, 2]
        Hs: int,
        Ws: int,
    ) -> torch.Tensor:
        """Sample x at offset positions using bilinear interpolation.

        Returns:
            [B, G, K, Hs, Ws, C//G]
        """
        B, C, H, W = x.shape
        G = self.groups
        K = self.K
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
        grid_base = torch.stack([grid_x, grid_y], dim=-1)  # [Hs, Ws, 2]
        grid_base = grid_base.unsqueeze(0).unsqueeze(3).unsqueeze(4)  # [1, Hs, Ws, 1, 1, 2]

        # Base kernel offsets in normalised coordinates [1, 1, 1, 1, K, 2]
        base_k = self.base_grid.to(x.device)  # [K, 2]
        base_k_norm = base_k.clone()
        base_k_norm[:, 0] /= W / 2
        base_k_norm[:, 1] /= H / 2
        base_k_norm = base_k_norm.view(1, 1, 1, 1, K, 2)  # [1, 1, 1, 1, K, 2]

        # Learned offsets (normalised by H/W)
        off_norm = offsets.clone()
        off_norm[..., 0] /= W / 2
        off_norm[..., 1] /= H / 2

        # Per-group sampling coordinates: broadcast [1,Hs,Ws,1,1,2] + [1,1,1,1,K,2]
        # + [B,Hs,Ws,G,K,2] → [B, Hs, Ws, G, K, 2]
        coords = grid_base + base_k_norm + off_norm
        coords = coords.clamp(-1, 1)

        # Sample x for each group separately.
        # Vectorise over K by treating B*K as the batch dimension for grid_sample.
        x_groups = x.reshape(B, G, Cg, H, W)  # [B, G, Cg, H, W]

        sampled = []
        for g in range(G):
            # Expand input along the K axis: [B, K, Cg, H, W] → [B*K, Cg, H, W]
            xg = x_groups[:, g]  # [B, Cg, H, W]
            xg = xg.unsqueeze(1).expand(-1, K, -1, -1, -1).reshape(B * K, Cg, H, W)

            # Sampling grid for group g: [B, Hs, Ws, K, 2]
            # .contiguous() is needed before reshape because indexing on dim 3 (G)
            # produces a non-contiguous slice.
            cg = coords[:, :, :, g, :, :].contiguous()  # [B, Hs, Ws, K, 2]
            cg = cg.permute(0, 3, 1, 2, 4).reshape(B * K, Hs, Ws, 2)  # [B*K, Hs, Ws, 2]

            out_g = F.grid_sample(
                xg, cg, mode="bilinear", align_corners=False, padding_mode="zeros"
            )
            # out_g: [B*K, Cg, Hs, Ws] → [B, K, Cg, Hs, Ws] → [B, Cg, Hs, Ws, K]
            out_g = out_g.reshape(B, K, Cg, Hs, Ws)
            out_g = out_g.permute(0, 2, 3, 4, 1)  # [B, Cg, Hs, Ws, K]
            sampled.append(out_g)

        # Stack groups: [B, G, Cg, Hs, Ws, K]
        out = torch.stack(sampled, dim=1)
        # Rearrange to [B, G, K, Hs, Ws, Cg]
        out = out.permute(0, 1, 5, 3, 4, 2)
        return out
