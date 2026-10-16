"""``PatchMerging``: Downsample by 2× and double channels: H×W×C → H/2×W/2×2C."""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchMerging(nn.Module):
    """Downsample by 2× and double channels: H×W×C → H/2×W/2×2C.

    Takes 2×2 spatial neighbours, concatenates channel-wise (→ 4C),
    applies LayerNorm, then projects 4C → 2C with a linear layer.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, H, W, C] → [B, H/2, W/2, 2C]"""
        B, H, W, C = x.shape
        # Sample every-other pixel in both spatial dimensions
        x0 = x[:, 0::2, 0::2, :]  # top-left
        x1 = x[:, 1::2, 0::2, :]  # bottom-left
        x2 = x[:, 0::2, 1::2, :]  # top-right
        x3 = x[:, 1::2, 1::2, :]  # bottom-right
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, H/2, W/2, 4C]
        return self.reduction(self.norm(x))
