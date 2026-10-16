"""``ConvNeXtV2Block``: Single ConvNeXt V2 residual block."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.cnn.convnext.grn import GRN
from hale_core.nn.vision.cnn.convnext.stochastic_depth import StochasticDepth


class ConvNeXtV2Block(nn.Module):
    """Single ConvNeXt V2 residual block.

    Sequence: DWConv → LayerNorm → Linear(4x expand) → GeLU → GRN → Linear(contract)
    with a residual connection.  All linear ops are implemented as pointwise 1×1
    convolutions via nn.Linear on the transposed tensor (channels-last) for clarity.

    Args:
        dim:        Number of input (and output) channels.
        drop_path:  Stochastic depth drop probability (0 = disabled).
    """

    def __init__(self, dim: int, drop_path: float = 0.0) -> None:
        super().__init__()

        # Depthwise 7×7 conv — processes spatial context within each channel
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        # LayerNorm in channels-last space (after transpose)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # Inverted bottleneck: expand 4× for more capacity
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        # GRN — the V2-specific feature competition layer
        self.grn = GRN(4 * dim)
        # Contract back to dim
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # Stochastic depth: randomly drops the entire block during training
        # (equivalent to randomly skipping the residual branch)
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.dwconv(x)  # [B, C, H, W] — depthwise spatial conv
        x = x.permute(0, 2, 3, 1)  # → [B, H, W, C] for LayerNorm + Linear
        x = self.norm(x)
        x = self.pwconv1(x)  # [B, H, W, 4C]
        x = self.act(x)
        x = self.grn(x)  # feature competition
        x = self.pwconv2(x)  # [B, H, W, C]
        x = x.permute(0, 3, 1, 2)  # → [B, C, H, W]

        return residual + self.drop_path(x)
