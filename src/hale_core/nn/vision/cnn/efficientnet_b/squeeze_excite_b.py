"""``SqueezeExciteB``: SE block for EfficientNet-B with SiLU activations."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExciteB(nn.Module):
    """SE block for EfficientNet-B with SiLU activations.

    Args:
        in_chs:       Input (and output) channel count.
        se_ratio:     Reduction ratio applied to in_chs.  Default 0.25 (paper: 0.25).
    """

    def __init__(self, in_chs: int, se_ratio: float = 0.25) -> None:
        super().__init__()
        squeezed = max(1, int(in_chs * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_chs, squeezed, kernel_size=1, bias=True)
        self.fc2 = nn.Conv2d(squeezed, in_chs, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`SqueezeExciteB`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        s = self.pool(x)
        s = F.silu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s
