"""``ResBottleneckBlock``: Standard RegNet block (bottleneck_ratio=1 so in==out width)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.cnn.regnet.squeeze_excite import SqueezeExcite


class ResBottleneckBlock(nn.Module):
    """Standard RegNet block (bottleneck_ratio=1 so in==out width).

    Args:
        in_channels:   Input channels.
        out_channels:  Output channels.
        stride:        Spatial stride (1 or 2).
        group_width:   Group width for the 3×3 grouped conv.
        use_se:        Whether to add Squeeze-Excite attention.
        se_reduction:  SE squeeze reduction factor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        group_width: int = 1,
        use_se: bool = False,
        se_reduction: int = 4,
    ) -> None:
        super().__init__()
        groups = max(1, out_channels // group_width)

        # 1×1 proj
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 3×3 grouped conv
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=stride, padding=1, groups=groups, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Channel attention (RegNetY only)
        self.se = SqueezeExcite(out_channels, se_reduction) if use_se else nn.Identity()

        # 1×1 expansion
        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Shortcut projection when shape changes
        self.shortcut: Optional[nn.Module] = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = x if self.shortcut is None else self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.se(out)
        out = self.bn3(self.conv3(out))

        return F.relu(out + skip, inplace=True)
