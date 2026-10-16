"""``SqueezeExcitation``: Channel-wise gating: global avg pool → 2-FC → sigmoid scale."""

from __future__ import annotations

import torch
import torch.nn as nn


class SqueezeExcitation(nn.Module):
    """Channel-wise gating: global avg pool → 2-FC → sigmoid scale.

    Args:
        channels:    Input/output channel count.
        rd_ratio:    Reduction ratio for bottleneck (default 0.25 = 4×).
        activation:  Activation between FC layers.
    """

    def __init__(
        self,
        channels: int,
        rd_ratio: float = 0.25,
        activation: nn.Module = None,
    ) -> None:
        super().__init__()
        rd = max(1, int(channels * rd_ratio))
        self.fc1 = nn.Linear(channels, rd, bias=True)
        self.fc2 = nn.Linear(rd, channels, bias=True)
        self.act = activation or nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] → scale [B, C, 1, 1] applied to x."""
        s = x.mean(dim=(2, 3))  # Global avg pool: [B, C]
        s = self.act(self.fc1(s))  # Bottleneck
        s = self.fc2(s).sigmoid()  # Gate in [0, 1]
        return x * s[:, :, None, None]  # Scale channels
