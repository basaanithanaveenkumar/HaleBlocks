"""``LayerNorm2d``: LayerNorm adapted for [B, C, H, W] spatial tensors."""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm adapted for [B, C, H, W] spatial tensors.

    Transposes to channels-last before normalising, then back to channels-first.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`LayerNorm2d`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        x = x.permute(0, 2, 3, 1)  # → [B, H, W, C]
        x = super().forward(x)
        return x.permute(0, 3, 1, 2)  # → [B, C, H, W]
