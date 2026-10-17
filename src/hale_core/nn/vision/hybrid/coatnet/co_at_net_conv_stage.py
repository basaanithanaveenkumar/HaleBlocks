"""``CoAtNetConvStage``: Sequence of MBConv blocks with optional downsampling at the start."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.hybrid.coatnet.mb_conv_block import MBConvBlock


class CoAtNetConvStage(nn.Module):
    """Sequence of MBConv blocks with optional downsampling at the start.

    Args:
        in_ch:    Input channels.
        out_ch:   Output channels.
        depth:    Number of MBConv blocks.
        stride:   Spatial stride for the first block (1 = no downscale).
        drop_path_rates: Per-block stochastic depth rates.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        depth: int,
        stride: int = 2,
        drop_path_rates: list | None = None,
    ) -> None:
        super().__init__()
        rates = drop_path_rates or [0.0] * depth
        # First block handles channel change + optional stride
        blocks = [MBConvBlock(in_ch, out_ch, stride=stride, drop_path=rates[0])]
        for i in range(1, depth):
            blocks.append(MBConvBlock(out_ch, out_ch, stride=1, drop_path=rates[i]))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`CoAtNetConvStage`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        return self.blocks(x)
