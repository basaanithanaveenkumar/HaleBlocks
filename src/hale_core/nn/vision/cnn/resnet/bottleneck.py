"""``Bottleneck``: Residual bottleneck block for ResNet-50/101/152/200."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class Bottleneck(nn.Module):
    """Residual bottleneck block for ResNet-50/101/152/200.

    Uses a 1×1 → 3×3 → 1×1 pattern (expansion=4), significantly reducing FLOPs
    compared to two stacked 3×3 convolutions when channel counts are large.

    Args:
        in_planes:  Number of input channels.
        planes:     "bottleneck width"; output channels = planes × expansion = planes × 4.
        stride:     Stride for the 3×3 conv; when >1, spatial downsampling occurs.
        downsample: Optional projection shortcut when input/output shapes differ.
    """

    expansion: int = 4  # output channels = planes * 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        out_channels = planes * self.expansion

        # 1×1 conv: reduce channels from in_planes to planes (the "bottleneck")
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 3×3 conv: main spatial processing; stride here for downsampling
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        # 1×1 conv: expand channels back to planes × 4
        self.conv3 = nn.Conv2d(planes, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`Bottleneck`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))  # no activation before add

        if self.downsample is not None:
            identity = self.downsample(x)

        out = F.relu(out + identity, inplace=True)
        return out
