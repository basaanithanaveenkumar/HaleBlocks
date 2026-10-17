"""``SEBlock``: Squeeze-and-Excitation channel attention block."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block.

    Reduces channels by reduction ratio r, then re-expands, producing a per-channel
    scaling vector in [0, 1] via Sigmoid. This vector multiplies the input features.

    Args:
        channels:  Input (and output) channel count.
        reduction: Squeeze ratio r.  Default 16 (original paper setting).
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        squeezed = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Two linear layers implemented as 1×1 convolutions (avoids reshape)
        self.fc1 = nn.Conv2d(channels, squeezed, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(squeezed, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`SEBlock`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        # Squeeze: global spatial average
        s = self.pool(x)  # [B, C, 1, 1]
        # Excitation: FC → ReLU → FC → Sigmoid
        s = F.relu(self.fc1(s), inplace=True)  # [B, C//r, 1, 1]
        s = torch.sigmoid(self.fc2(s))  # [B, C,    1, 1]
        # Scale: channel-wise multiplication
        return x * s
