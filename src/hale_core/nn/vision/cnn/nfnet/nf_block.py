"""``NFBlock``: Normalizer-Free residual block with variance preservation."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.cnn.nfnet.squeeze_excite import SqueezeExcite
from hale_core.nn.vision.cnn.nfnet.stochastic_depth import StochasticDepth
from hale_core.nn.vision.cnn.nfnet.ws_conv2d import WSConv2d


class NFBlock(nn.Module):
    """Normalizer-Free residual block with variance preservation.

    Uses the formula: y = x + alpha * F(x / beta)
    where alpha (scalar, ≈0.2) scales the residual branch to prevent variance
    accumulation, and beta = expected_std_of_x normalises the input.

    Args:
        in_channels:   Input channel count.
        out_channels:  Output channel count.
        stride:        Spatial stride for the 3×3 conv.
        group_size:    Group width for grouped conv.
        beta:          Expected input standard deviation (accumulated across blocks).
        alpha:         Residual branch scale factor.
        squeeze_ratio: SE squeeze ratio (0 = no SE).
        drop_path:     Stochastic depth rate.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        group_size: int = 128,
        beta: float = 1.0,
        alpha: float = 0.2,
        squeeze_ratio: float = 0.5,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.beta = beta
        # Learnable alpha per block (init to provided alpha value)
        self.alpha = nn.Parameter(torch.full([], alpha))

        groups = max(1, out_channels // group_size)
        # Bottleneck: in → out → out (groups) → out
        self.conv1 = WSConv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = WSConv2d(
            out_channels, out_channels, 3, stride=stride, padding=1, groups=groups, bias=False
        )
        self.conv3 = WSConv2d(out_channels, out_channels, 1, bias=False)

        self.se = SqueezeExcite(out_channels, squeeze_ratio) if squeeze_ratio > 0 else nn.Identity()
        self.drop_path = StochasticDepth(drop_path) if drop_path > 0 else nn.Identity()

        # Skip connection if shape changes
        self.shortcut: Optional[nn.Module] = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = WSConv2d(in_channels, out_channels, 1, stride=stride, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalise input variance before the residual branch
        x_norm = x / self.beta

        out = F.gelu(self.conv1(x_norm))
        out = F.gelu(self.conv2(out))
        out = self.conv3(out)
        out = self.se(out)
        out = self.drop_path(out) * self.alpha

        skip = x if self.shortcut is None else self.shortcut(x)
        return out + skip
