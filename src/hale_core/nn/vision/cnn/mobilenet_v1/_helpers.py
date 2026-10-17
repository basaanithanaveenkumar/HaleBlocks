"""Shared helpers of :mod:`hale_core.nn.vision.cnn.mobilenet_v1` (MobileNetV1 Vision Backbones)."""

from __future__ import annotations

import torch
import torch.nn as nn


class _DWSConv(nn.Module):
    """Depthwise Separable Convolution block with BN+ReLU after each conv.

    Factorises a standard k×k conv into:
      1. Depthwise k×k (groups=in_chs)  — spatial mixing per channel
      2. Pointwise 1×1                  — cross-channel mixing

    Args:
        in_chs:  Input channels.
        out_chs: Output channels.
        stride:  Spatial stride applied to the depthwise conv.
    """

    def __init__(self, in_chs: int, out_chs: int, stride: int = 1) -> None:
        super().__init__()
        pad = 1
        self.dw = nn.Sequential(
            # Depthwise: each input channel filtered independently
            nn.Conv2d(
                in_chs, in_chs, kernel_size=3, stride=stride, padding=pad, groups=in_chs, bias=False
            ),
            nn.BatchNorm2d(in_chs),
            nn.ReLU(inplace=True),
        )
        self.pw = nn.Sequential(
            # Pointwise: 1×1 cross-channel projection
            nn.Conv2d(in_chs, out_chs, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_chs),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`_DWSConv`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        return self.pw(self.dw(x))
