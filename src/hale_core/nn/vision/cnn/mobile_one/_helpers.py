"""Shared helpers of :mod:`hale_core.nn.vision.cnn.mobile_one` (MobileOne — Re-Parameterisable Efficient Backbone (Vasu et al. 2022, Apple Research))."""

from __future__ import annotations

import torch
import torch.nn as nn


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
    w = conv.weight  # [C_out, C_in/G, kH, kW]
    mean = bn.running_mean  # [C_out]
    var = bn.running_var  # [C_out]
    beta = bn.bias  # [C_out]
    gamma = bn.weight  # [C_out]
    eps = bn.eps

    scale = gamma / (var + eps).sqrt()  # [C_out]
    # reshape scale for broadcasting over spatial dims
    w_fused = w * scale.view(-1, 1, 1, 1)
    b_fused = beta - mean * scale
    return w_fused, b_fused


class _ConvBN(nn.Sequential):
    """Conv2d → BatchNorm2d (no activation)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        pad = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=pad,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )


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


def _make_divisible(v: float, divisor: int) -> int:
    """Round v up to the nearest multiple of divisor (min divisor)."""
    return max(divisor, int(v + divisor / 2) // divisor * divisor)
