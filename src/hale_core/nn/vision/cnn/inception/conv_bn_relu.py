"""``ConvBnRelu``: Conv2d + BatchNorm2d + ReLU — the atomic unit in all Inception variants."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    """Conv2d + BatchNorm2d + ReLU — the atomic unit in all Inception variants."""

    def __init__(self, in_ch: int, out_ch: int, **conv_kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, bias=False, **conv_kwargs)
        self.bn = nn.BatchNorm2d(out_ch, eps=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`ConvBnRelu`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        return F.relu(self.bn(self.conv(x)), inplace=True)
