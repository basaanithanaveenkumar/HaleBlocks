"""``MBConvB``: Mobile Inverted Bottleneck Conv for EfficientNet-B series."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.cnn.efficientnet_b._helpers import _make_divisible
from hale_core.nn.vision.cnn.efficientnet_b.squeeze_excite_b import SqueezeExciteB
from hale_core.nn.vision.cnn.efficientnet_b.stochastic_depth import StochasticDepth


class MBConvB(nn.Module):
    """Mobile Inverted Bottleneck Conv for EfficientNet-B series.

    When expand_ratio=1 the expansion 1×1 conv is skipped (no channel change).
    SE is applied to the expanded (depthwise) channel count, not the input count.
    This differs from SE-ResNet where SE is applied to the bottleneck output.

    Args:
        in_chs:       Input channels.
        out_chs:      Output channels.
        stride:       Spatial stride (1 or 2).
        expand_ratio: Channel expansion before depthwise (typically 6, or 1 for first stage).
        kernel_size:  Depthwise conv kernel size (3 or 5).
        se_ratio:     SE reduction ratio relative to expanded channels.
        drop_path:    Stochastic depth probability.
    """

    def __init__(
        self,
        in_chs: int,
        out_chs: int,
        stride: int = 1,
        expand_ratio: int = 6,
        kernel_size: int = 3,
        se_ratio: float = 0.25,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        mid_chs = _make_divisible(in_chs * expand_ratio, 8)
        has_skip = stride == 1 and in_chs == out_chs
        pad = (kernel_size - 1) // 2

        layers: list[nn.Module] = []

        # Expansion phase (skip when ratio=1 — nothing changes channel-wise)
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_chs, mid_chs, kernel_size=1, bias=False),
                nn.BatchNorm2d(mid_chs),
                nn.SiLU(inplace=True),
            ]
        else:
            mid_chs = in_chs  # no expansion

        # Depthwise conv
        layers += [
            nn.Conv2d(
                mid_chs,
                mid_chs,
                kernel_size=kernel_size,
                stride=stride,
                padding=pad,
                groups=mid_chs,
                bias=False,
            ),
            nn.BatchNorm2d(mid_chs),
            nn.SiLU(inplace=True),
        ]

        # Squeeze-Excite on expanded channels
        if se_ratio > 0.0:
            layers.append(SqueezeExciteB(mid_chs, se_ratio=se_ratio))

        # Projection
        layers += [
            nn.Conv2d(mid_chs, out_chs, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_chs),
        ]

        self.conv = nn.Sequential(*layers)
        self.drop_path = StochasticDepth(drop_path) if has_skip else nn.Identity()
        self.has_skip = has_skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`MBConvB`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        if self.has_skip:
            return x + self.drop_path(self.conv(x))
        return self.conv(x)
