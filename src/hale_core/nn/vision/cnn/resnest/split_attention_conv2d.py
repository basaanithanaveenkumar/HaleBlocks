"""``SplitAttentionConv2d``: Convolution with split-attention (radix-wise gating)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SplitAttentionConv2d(nn.Module):
    """Convolution with split-attention (radix-wise gating).

    Produces `radix` feature map splits, softmax-weights them using a global
    attention branch, then returns the weighted sum.

    Args:
        in_channels:  Input channel count.
        out_channels: Output channel count (per split, before combining).
        kernel_size:  Conv kernel size.
        stride:       Spatial stride.
        padding:      Spatial padding.
        groups:       Number of convolution groups (cardinality).
        radix:        Number of splits (default 2).
        reduction:    Reduction factor for the attention FC bottleneck.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
        radix: int = 2,
        reduction: int = 4,
    ) -> None:
        super().__init__()
        inter_channels = max(in_channels * radix // reduction, 32)
        self.radix = radix
        self.groups = groups
        self.out_ch = out_channels

        # Main conv: produces out_channels × radix feature maps
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * radix,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups * radix,
            bias=False,
        )
        self.bn0 = nn.BatchNorm2d(out_channels * radix)

        # Attention branch: globalpool → BN → FC → ReLU → FC → softmax
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.fc1 = nn.Conv2d(out_channels, inter_channels, kernel_size=1, groups=groups, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_channels)
        self.fc2 = nn.Conv2d(
            inter_channels, out_channels * radix, kernel_size=1, groups=groups, bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Produce all radix feature maps
        x = F.relu(self.bn0(self.conv(x)), inplace=True)  # [B, C*radix, H, W]
        B, CR, H, W = x.shape
        C = self.out_ch

        if self.radix > 1:
            # Split into radix branches: [B, radix, C, H, W]
            splits = x.reshape(B, self.radix, C, H, W)
            # Sum over radix for the attention query
            gap = splits.sum(dim=1)  # [B, C, H, W]
        else:
            gap = x

        # Attention branch
        gap = self.bn1(gap)
        gap = self.pool(gap)  # [B, C, 1, 1]
        gap = F.relu(self.bn2(self.fc1(gap)), inplace=True)  # [B, inter, 1, 1]
        atten = self.fc2(gap)  # [B, C*radix, 1, 1]

        if self.radix > 1:
            # Softmax over radix axis
            atten = atten.reshape(B, self.radix, C, 1, 1)
            atten = F.softmax(atten, dim=1)  # [B, radix, C, 1, 1]
            # Weighted combination
            out = (splits * atten).sum(dim=1)  # [B, C, H, W]
        else:
            out = x * torch.sigmoid(atten)

        return out.contiguous()
