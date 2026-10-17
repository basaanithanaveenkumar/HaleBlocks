"""``MBConvBlock``: Mobile Inverted Bottleneck Convolution block."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.hybrid.coatnet.drop_path import DropPath
from hale_core.nn.vision.hybrid.coatnet.squeeze_excitation import SqueezeExcitation


class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution block.

    Used in S1 and S2 of CoAtNet. Pattern:
      [input]
        ↓ (BatchNorm — pre-norm applied to input)
        ↓ expand 1×1 conv (expand_ratio × C channels)
        ↓ depthwise 3×3 conv (groups=out_channels for true depthwise)
        ↓ Squeeze-Excitation
        ↓ project 1×1 conv (back to out_channels)
        ↓ (+ residual if in_channels == out_channels and stride==1)

    Args:
        in_ch:        Input channels.
        out_ch:       Output channels.
        stride:       Spatial stride (1 or 2).
        expand_ratio: Expansion multiplier for intermediate channels.
        se_ratio:     SE reduction ratio.
        drop_path:    Stochastic depth rate.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        expand_ratio: int = 4,
        se_ratio: float = 0.25,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.use_residual = in_ch == out_ch and stride == 1

        # Pre-BatchNorm normalises the input before the block
        self.pre_norm = nn.BatchNorm2d(in_ch)

        self.expand = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.expand_act = nn.SiLU()

        # Depthwise 3×3 — groups=mid_ch means each channel has its own filter
        self.dw_conv = nn.Conv2d(
            mid_ch, mid_ch, 3, stride=stride, padding=1, groups=mid_ch, bias=False
        )
        self.dw_bn = nn.BatchNorm2d(mid_ch)
        self.dw_act = nn.SiLU()

        self.se = SqueezeExcitation(mid_ch, rd_ratio=se_ratio / expand_ratio)

        self.project = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_ch)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`MBConvBlock`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        residual = x
        x = self.pre_norm(x)
        x = self.expand_act(self.expand(x))
        x = self.dw_act(self.dw_bn(self.dw_conv(x)))
        x = self.se(x)
        x = self.project_bn(self.project(x))
        if self.use_residual:
            x = residual + self.drop_path(x)
        return x
