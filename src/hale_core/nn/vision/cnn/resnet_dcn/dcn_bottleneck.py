"""``DCNBottleneck``: Bottleneck block with DeformConv2d replacing the 3×3 standard conv."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.cnn.resnet_dcn.deform_conv2d import DeformConv2d


class DCNBottleneck(nn.Module):
    """Bottleneck block with DeformConv2d replacing the 3×3 standard conv.

    Identical to Bottleneck except conv2 is a DeformConv2d.  The 1×1 squeeze
    and 1×1 expand convolutions remain standard — they operate on single
    spatial locations and so don't benefit from deformability.

    Args:
        in_planes:  Input channel count.
        planes:     Bottleneck width (output = planes × 4).
        stride:     Spatial stride (downsampling) for the deformable 3×3 conv.
        downsample: Optional projection shortcut.
    """

    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        out_channels = planes * self.expansion

        # 1×1 squeeze
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # DCNv2 replaces the 3×3 standard conv
        self.conv2 = DeformConv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(planes)

        # 1×1 expand
        self.conv3 = nn.Conv2d(planes, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`DCNBottleneck`.

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
