"""``ResNeXtBottleneck``: ResNeXt Bottleneck with grouped (cardinality) convolution."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNeXtBottleneck(nn.Module):
    """ResNeXt Bottleneck with grouped (cardinality) convolution.

    The key difference from standard Bottleneck: the 3×3 conv uses groups=C
    (cardinality), which implements C parallel transformations of equal width.
    Mathematically equivalent to a width-C grouped conv but much more expressive
    than a single wider conv at the same parameter budget.

    Args:
        in_planes:   Input channels.
        planes:      Stage "planes" (output = planes * expansion = planes * 4).
        stride:      Stride for the 3×3 grouped conv.
        cardinality: Number of transformation groups (default 32).
        base_width:  Width per group in the bottleneck (default 4).
        downsample:  Optional projection shortcut.
    """

    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        cardinality: int = 32,
        base_width: int = 4,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        # Bottleneck width = C * D; at the standard stage channel counts this
        # gives much wider intermediate features than ResNet bottleneck at similar cost
        mid_ch = int(planes * (base_width / 64.0)) * cardinality

        self.conv1 = nn.Conv2d(in_planes, mid_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_ch)

        # Grouped conv: groups=cardinality implements the aggregated transformation
        self.conv2 = nn.Conv2d(
            mid_ch, mid_ch, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_ch)

        out_ch = planes * self.expansion
        self.conv3 = nn.Conv2d(mid_ch, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`ResNeXtBottleneck`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        identity = x
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return F.relu(out + identity, inplace=True)
