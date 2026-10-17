"""``SqueezeExcite``: Channel attention: global-pool → squeeze linear → GELU → excite linear →
sigmoid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite(nn.Module):
    """Channel attention: global-pool → squeeze linear → GELU → excite linear → sigmoid."""

    def __init__(self, in_channels: int, squeeze_ratio: float = 0.5) -> None:
        super().__init__()
        mid = max(1, int(in_channels * squeeze_ratio))
        self.squeeze = nn.Linear(in_channels, mid)
        self.excite = nn.Linear(mid, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`SqueezeExcite`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        # x: [B, C, H, W]
        scale = x.mean(dim=[2, 3])  # [B, C] global average pool
        scale = F.gelu(self.squeeze(scale))  # [B, mid]
        scale = torch.sigmoid(self.excite(scale))  # [B, C]
        return x * scale.unsqueeze(-1).unsqueeze(-1)
