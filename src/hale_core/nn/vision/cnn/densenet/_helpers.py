"""Shared helpers of :mod:`hale_core.nn.vision.cnn.densenet` (DenseNet Vision Backbones)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DenseLayer(nn.Module):
    """Single layer in a DenseBlock.

    Implements the BN-ReLU-Conv1×1-BN-ReLU-Conv3×3 "bottleneck" design.
    Input is ALL prior layer outputs concatenated; output is k new channels.

    The 1×1 conv compresses the accumulated channels to 4k before the 3×3 conv,
    keeping computation tractable as the concatenation dimension grows.

    Args:
        num_input_features: Number of input channels (all prior features).
        growth_rate:        k — number of new channels this layer produces.
        memory_efficient:   Use gradient checkpointing (not implemented here).
    """

    def __init__(self, num_input_features: int, growth_rate: int) -> None:
        super().__init__()
        inter_ch = 4 * growth_rate  # 1×1 conv target = 4k

        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.conv1 = nn.Conv2d(num_input_features, inter_ch, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(inter_ch)
        self.conv2 = nn.Conv2d(inter_ch, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`_DenseLayer`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        # Typical implementation receives a list of all previous outputs which
        # are concatenated inside forward — this matches DenseNet paper appendix.
        out = self.conv1(F.relu(self.norm1(x), inplace=True))
        out = self.conv2(F.relu(self.norm2(out), inplace=True))
        return out


class _Transition(nn.Module):
    """Transition layer between DenseBlocks.

    Halves both channel count (via 1×1 conv) and spatial resolution (AvgPool).
    The compression factor of 0.5 is fixed per the original paper (θ=0.5).

    Args:
        num_input_features:  Input channels (= DenseBlock output channels).
        num_output_features: Output channels (= num_input_features // 2).
    """

    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`_Transition`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        return self.pool(self.conv(F.relu(self.norm(x), inplace=True)))
