"""``LayerNorm2d``: LayerNorm adapted for channels-first (B, C, H, W) spatial tensors."""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm adapted for channels-first (B, C, H, W) spatial tensors.

    PyTorch's nn.LayerNorm expects the normalised dims to be the last dims.
    This wrapper transposes to channels-last before normalising, then transposes back.
    This is the standard approach used by ConvNeXt and many modern CNNs.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        x = x.permute(0, 2, 3, 1)  # → [B, H, W, C]
        x = super().forward(x)  # LayerNorm over last dim C
        return x.permute(0, 3, 1, 2)  # → [B, C, H, W]
