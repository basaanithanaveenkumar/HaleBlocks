"""``SqueezeExcite``: Channel attention via global pooling followed by two fully-connected layers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite(nn.Module):
    """Channel attention via global pooling followed by two fully-connected layers.

    SE reduces channels by se_ratio, then re-expands to original count.
    Output is used to SCALE the input features (not add to them).

    Args:
        in_chs:   Input channel count.
        se_ratio: Reduction ratio for the bottleneck fc layer (default 0.25 → C//4).
    """

    def __init__(self, in_chs: int, se_ratio: float = 0.25) -> None:
        super().__init__()
        se_chs = max(1, int(in_chs * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_chs, se_chs, kernel_size=1)
        self.fc2 = nn.Conv2d(se_chs, in_chs, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`SqueezeExcite`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        scale = self.pool(x)
        scale = F.silu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale
