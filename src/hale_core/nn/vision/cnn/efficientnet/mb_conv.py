"""``MBConv``: Standard Mobile Inverted Bottleneck Conv for EfficientNetV2."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.cnn.efficientnet._helpers import _make_divisible
from hale_core.nn.vision.cnn.efficientnet.squeeze_excite import SqueezeExcite
from hale_core.nn.vision.cnn.efficientnet.stochastic_depth import StochasticDepth


class MBConv(nn.Module):
    """Standard Mobile Inverted Bottleneck Conv for EfficientNetV2.

    Separate expand 1×1 + depthwise k×k + project 1×1 with SE.

    Args:
        in_chs:        Input channels.
        out_chs:       Output channels.
        stride:        Spatial stride.
        expand_ratio:  Channel expansion factor.
        kernel_size:   Depthwise kernel size (3 or 5).
        se_ratio:      Squeeze-Excite reduction ratio.
        drop_path:     Stochastic depth probability.
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

        self.conv = nn.Sequential(
            # Expand
            nn.Conv2d(in_chs, mid_chs, 1, bias=False),
            nn.BatchNorm2d(mid_chs),
            nn.SiLU(inplace=True),
            # Depthwise
            nn.Conv2d(
                mid_chs,
                mid_chs,
                kernel_size,
                stride=stride,
                padding=pad,
                groups=mid_chs,
                bias=False,
            ),
            nn.BatchNorm2d(mid_chs),
            nn.SiLU(inplace=True),
            # Squeeze-Excite
            SqueezeExcite(mid_chs, se_ratio),
            # Project
            nn.Conv2d(mid_chs, out_chs, 1, bias=False),
            nn.BatchNorm2d(out_chs),
        )
        self.stochastic_depth = StochasticDepth(drop_path) if has_skip else nn.Identity()
        self.has_skip = has_skip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`MBConv`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        if self.has_skip:
            return x + self.stochastic_depth(self.conv(x))
        return self.conv(x)
