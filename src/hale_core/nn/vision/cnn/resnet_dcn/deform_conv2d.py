"""``DeformConv2d``: Deformable Convolution v2 implemented via grid_sample."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeformConv2d(nn.Module):
    """Deformable Convolution v2 implemented via grid_sample.

    For each output spatial location, this layer:
      1. Predicts 2*k*k offsets (Δy, Δx per kernel grid point) via `offset_conv`.
      2. Predicts k*k modulation weights (sigmoid) via `mask_conv`.
      3. Samples input at offset positions using bilinear interpolation.
      4. Applies the learned kernel weights to the sampled features.

    The implementation uses F.grid_sample which handles out-of-bounds inputs
    by zero-padding, matching the typical CUDA DCN extension behaviour.

    Args:
        in_channels:  Number of input feature map channels.
        out_channels: Number of output feature map channels.
        kernel_size:  Spatial kernel size (typically 3).
        stride:       Stride of the convolution.
        padding:      Zero-padding applied to input (typically kernel_size//2).
        bias:         Whether to add a bias term to the output.

    Shape:
        Input:  [B, in_channels, H, W]
        Output: [B, out_channels, H', W']  where H', W' depend on stride/padding.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.k2 = kernel_size * kernel_size  # number of kernel grid positions

        # Offset predictor: 2 × k² channels (Δy and Δx for each grid point)
        # Zero-initialized so the deformation starts as identity (regular conv)
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * self.k2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        # Modulation predictor: k² channels per spatial location, sigmoid → [0,1]
        self.mask_conv = nn.Conv2d(
            in_channels,
            self.k2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        # Init mask weights to small values so masks start near 0.5 (sigmoid(0)=0.5)
        nn.init.zeros_(self.mask_conv.weight)
        nn.init.zeros_(self.mask_conv.bias)

        # Main convolution weight (applied to sampled features)
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, kernel_size, kernel_size))
        self.bias_param = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # Standard Kaiming init for conv weights
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Deformable conv2d forward pass.

        Strategy:
          - Predict per-location offsets and masks from input x.
          - Build a sampling grid of shape [B, H_out*W_out*k², 2] by adding
            offsets to the base regular grid.
          - Use grid_sample to gather bilinearly-interpolated samples.
          - Apply modulation weights and sum over kernel positions.
          - Finally apply the weight matrix to produce output channels.
        """
        B, C, H, W = x.shape

        # Predict offsets: [B, 2*k², H_out, W_out]
        offsets = self.offset_conv(x)
        # Predict masks: [B, k², H_out, W_out]
        masks = torch.sigmoid(self.mask_conv(x))

        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        # ── Build base regular-grid sampling coordinates ──────────────────────
        # Output positions in input-pixel space, accounting for stride
        out_y = torch.arange(H_out, device=x.device, dtype=x.dtype) * self.stride - self.padding
        out_x = torch.arange(W_out, device=x.device, dtype=x.dtype) * self.stride - self.padding

        # Kernel offsets relative to center (e.g. for k=3: -1, 0, +1 in each dim)
        ky_range = (
            torch.arange(self.kernel_size, device=x.device, dtype=x.dtype) - self.kernel_size // 2
        )
        kx_range = (
            torch.arange(self.kernel_size, device=x.device, dtype=x.dtype) - self.kernel_size // 2
        )

        # Base grid: [H_out, W_out, k, k, 2]  (y and x for each kernel position)
        base_y = out_y.view(H_out, 1, 1, 1) + ky_range.view(1, 1, self.kernel_size, 1)
        base_x = out_x.view(1, W_out, 1, 1) + kx_range.view(1, 1, 1, self.kernel_size)

        # Reshape to [H_out, W_out, k²]
        base_y = base_y.expand(H_out, W_out, self.kernel_size, self.kernel_size).reshape(
            H_out, W_out, self.k2
        )
        base_x = base_x.expand(H_out, W_out, self.kernel_size, self.kernel_size).reshape(
            H_out, W_out, self.k2
        )

        # ── Add learned offsets ───────────────────────────────────────────────
        # offsets shape: [B, 2*k², H_out, W_out] → split into dy, dx
        dy = offsets[:, : self.k2, :, :].permute(0, 2, 3, 1)  # [B, H_out, W_out, k²]
        dx = offsets[:, self.k2 :, :, :].permute(0, 2, 3, 1)  # [B, H_out, W_out, k²]

        # Absolute sampling positions in input-pixel space
        sample_y = base_y.unsqueeze(0) + dy  # [B, H_out, W_out, k²]
        sample_x = base_x.unsqueeze(0) + dx  # [B, H_out, W_out, k²]

        # ── Normalise coordinates to [-1, 1] for grid_sample ─────────────────
        # grid_sample expects x,y in [-1, 1] where (-1,-1) = top-left corner
        norm_y = 2.0 * sample_y / (H - 1) - 1.0
        norm_x = 2.0 * sample_x / (W - 1) - 1.0

        # Stack into grid: [B, H_out, W_out*k², 2] (x first for grid_sample)
        # We sample all H_out*W_out*k² points in one grid_sample call
        grid = torch.stack([norm_x, norm_y], dim=-1)  # [B, H_out, W_out, k², 2]
        grid = grid.view(B, H_out, W_out * self.k2, 2)  # [B, H_out, W_out*k², 2]

        # ── Bilinear sampling ─────────────────────────────────────────────────
        # x: [B, C, H, W] → sample at grid positions → [B, C, H_out, W_out*k²]
        sampled = F.grid_sample(x, grid, mode="bilinear", align_corners=True, padding_mode="zeros")
        sampled = sampled.view(B, C, H_out, W_out, self.k2)  # [B, C, H_out, W_out, k²]
        sampled = sampled.permute(0, 2, 3, 4, 1)  # [B, H_out, W_out, k², C]

        # ── Apply modulation masks ────────────────────────────────────────────
        masks = masks.permute(0, 2, 3, 1).unsqueeze(-1)  # [B, H_out, W_out, k², 1]
        sampled = sampled * masks  # broadcast over C

        # ── Apply conv weights (einsum over k² kernel positions and C) ────────
        # weight: [out_channels, C, k, k] → reshape to [out_channels, C, k²]
        w = self.weight.view(self.out_channels, C, self.k2)  # [O, C, k²]

        # sampled: [B, H_out, W_out, k², C] → sum over k² and C
        # out[b,h,w,o] = sum_{p,c} sampled[b,h,w,p,c] * w[o,c,p]
        out = torch.einsum("bhwpc,ocp->bhwo", sampled, w)  # [B, H_out, W_out, O]
        out = out.permute(0, 3, 1, 2)  # [B, O, H_out, W_out]

        if self.bias_param is not None:
            out = out + self.bias_param.view(1, -1, 1, 1)

        return out
