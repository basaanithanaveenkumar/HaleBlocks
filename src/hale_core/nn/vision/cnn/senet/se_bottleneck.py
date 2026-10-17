"""``SEBottleneck``: ResNet Bottleneck with SE channel attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.cnn.senet.se_block import SEBlock


class SEBottleneck(nn.Module):
    """ResNet Bottleneck with SE channel attention.

    Architecture:
        input → Conv1×1 → BN → ReLU
              → Conv3×3 → BN → ReLU           (groups for ResNeXt variant)
              → Conv1×1 → BN
              → SE(reduction=16)               ← channel attention here
              → + residual → ReLU

    Args:
        in_planes:  Input channels.
        planes:     Bottleneck width (output = planes × 4).
        stride:     Applied to the 3×3 conv for spatial downsampling.
        downsample: 1×1 projection shortcut for shape matching.
        groups:     Convolution groups for the 3×3 (1=ResNet, 32=ResNeXt).
        base_width: Base channel width per group for ResNeXt scaling.
        reduction:  SE reduction ratio.
    """

    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        reduction: int = 16,
    ) -> None:
        super().__init__()
        # ResNeXt-style width scaling: actual mid-channels depend on groups + base_width
        width = int(planes * (base_width / 64.0)) * groups
        out_channels = planes * self.expansion

        # 1×1 reduce
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)

        # 3×3 spatial (with optional grouping for ResNeXt)
        self.conv2 = nn.Conv2d(
            width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False
        )
        self.bn2 = nn.BatchNorm2d(width)

        # 1×1 expand
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # SE block applied to the expanded feature map
        self.se = SEBlock(out_channels, reduction=reduction)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`SEBottleneck`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))  # no activation before SE+add

        # SE recalibration before residual addition
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = F.relu(out + identity, inplace=True)
        return out
