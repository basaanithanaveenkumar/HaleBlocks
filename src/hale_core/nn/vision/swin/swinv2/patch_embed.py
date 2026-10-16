"""``PatchEmbed``: 4×4 non-overlapping patch projection (stem)."""

from __future__ import annotations

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """4×4 non-overlapping patch projection (stem).

    Converts [B, 3, H, W] → [B, H/4, W/4, embed_dim].
    Uses a single Conv2d(kernel=4, stride=4) followed by LayerNorm.
    """

    def __init__(self, in_chans: int = 3, embed_dim: int = 96) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] → [B, H/4, W/4, embed_dim]"""
        x = self.proj(x)  # [B, embed_dim, H/4, W/4]
        x = x.permute(0, 2, 3, 1)  # [B, H/4, W/4, embed_dim] — channels-last
        return self.norm(x)
