"""``InternImageDownsample``: Between-stage downsampling: LayerNorm → 3×3 stride-2 conv → double channels."""

from __future__ import annotations

import torch
import torch.nn as nn


class InternImageDownsample(nn.Module):
    """Between-stage downsampling: LayerNorm → 3×3 stride-2 conv → double channels."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(in_ch, eps=1e-6)
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`InternImageDownsample`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        # x: [B, C, H, W]
        x = x.permute(0, 2, 3, 1)  # → [B, H, W, C]
        x = self.norm(x).permute(0, 3, 1, 2)  # → [B, C, H, W]
        return self.conv(x)
