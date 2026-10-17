"""``WSConv2d``: Conv2d with Scaled Weight Standardisation (Brock et al. 2021, Sec. 3.1)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        """Forward pass of :class:`WSConv2d`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        w = self.weight
        # Subtract per-filter mean (dims: in_C, kH, kW)
        w = w - w.mean(dim=[1, 2, 3], keepdim=True)
        # Normalise variance — eps avoids div-by-zero for zero-weight filters
        w = w / (w.var(dim=[1, 2, 3], keepdim=True, unbiased=False) + 1e-5).sqrt()
        # Scale to maintain expected variance: gain / sqrt(fan_in)
        fan_in = w[0].numel()
        w = w * (self.gain / math.sqrt(fan_in))
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
