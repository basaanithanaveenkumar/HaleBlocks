"""``ResNeStBottleneck``: ResNeSt Bottleneck: 1×1 → SplitAttn3×3 → 1×1 with residual."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.cnn.resnest.split_attention_conv2d import SplitAttentionConv2d


class ResNeStBottleneck(nn.Module):
    """ResNeSt Bottleneck: 1×1 → SplitAttn3×3 → 1×1 with residual.

    Replaces the standard 3×3 conv of a ResNet Bottleneck with SplitAttentionConv2d.
    The 1×1 reduce and 1×1 expand convolutions are unchanged from ResNet.

    Args:
        in_planes:  Input channels.
        planes:     Bottleneck width (output = planes × 4).
        stride:     Stride for the split-attention conv.
        downsample: Projection shortcut (if needed).
        radix:      Number of split-attention branches.
        groups:     Convolution groups.
    """

    expansion: int = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        radix: int = 2,
        groups: int = 1,
    ) -> None:
        super().__init__()
        out_channels = planes * self.expansion

        # 1×1 reduce
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        # 3×3 split-attention (core innovation)
        self.conv2 = SplitAttentionConv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=groups,
            radix=radix,
        )

        # 1×1 expand
        self.conv3 = nn.Conv2d(planes, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`ResNeStBottleneck`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.conv2(out)  # split-attention (already includes BN+ReLU)
        out = F.relu(out, inplace=True)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = F.relu(out + identity, inplace=True)
        return out
