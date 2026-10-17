"""``BasicBlock``: Residual block for ResNet-18/34."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Residual block for ResNet-18/34.

    Uses two 3×3 convolutions without channel bottlenecking (expansion=1).
    This design is cheaper per block but stacks more layers to reach target capacity.

    Args:
        in_planes:  Number of input channels.
        planes:     Number of output channels (== in_planes × expansion = in_planes).
        stride:     Stride for the first conv; when >1, spatial downsampling occurs.
        downsample: Optional projection shortcut when input/output shapes differ.
    """

    expansion: int = 1  # output channels = planes * expansion

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        # First 3×3 conv; stride here performs spatial downsampling if stride>1
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # Second 3×3 conv; always stride=1 to preserve resolution after first conv
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # Shortcut: may be None (identity) or a 1×1 projection to match dims
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`BasicBlock`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        # Add shortcut BEFORE final activation — core residual learning formula
        if self.downsample is not None:
            identity = self.downsample(x)

        out = F.relu(out + identity, inplace=True)
        return out
