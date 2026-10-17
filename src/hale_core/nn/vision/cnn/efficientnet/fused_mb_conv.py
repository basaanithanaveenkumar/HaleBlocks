"""``FusedMBConv``: Fused Mobile Inverted Bottleneck Convolution."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.cnn.efficientnet._helpers import _make_divisible
from hale_core.nn.vision.cnn.efficientnet.squeeze_excite import SqueezeExcite
from hale_core.nn.vision.cnn.efficientnet.stochastic_depth import StochasticDepth


class FusedMBConv(nn.Module):
    """Fused Mobile Inverted Bottleneck Convolution.

    Replaces expand_1x1 + depthwise_3x3 with a single fused_3x3 convolution.
    Faster on modern accelerators for small-channel early stages.

    Args:
        in_chs:        Input channels.
        out_chs:       Output channels.
        stride:        Spatial stride.
        expand_ratio:  Channel expansion factor before projection.
        se_ratio:      Squeeze-Excite reduction ratio (0 = no SE).
        drop_path:     Stochastic depth probability.
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        stride: int = 1,
        expand_ratio: int = 1,
        se_ratio: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        mid_chs = _make_divisible(in_chs * expand_ratio, 8)
        has_skip = stride == 1 and in_chs == out_chs

        layers: list[nn.Module] = []

        if expand_ratio != 1:
            # Expand conv (fused 3×3 instead of 1×1 + depthwise)
            layers += [
                nn.Conv2d(in_chs, mid_chs, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(mid_chs),
                nn.SiLU(inplace=True),
            ]
        else:
            mid_chs = in_chs

        if se_ratio > 0.0:
            layers.append(SqueezeExcite(mid_chs, se_ratio))

        # Project conv (1×1)
        layers += [
            nn.Conv2d(mid_chs, out_chs, 1, bias=False),
            nn.BatchNorm2d(out_chs),
        ]
        if expand_ratio != 1:
            # Only add SiLU after project when no expansion (r=1 case)
            pass

        self.conv = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(drop_path) if has_skip else nn.Identity()
        self.has_skip = has_skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`FusedMBConv`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        if self.has_skip:
            return x + self.stochastic_depth(self.conv(x))
        return self.conv(x)
