"""``MaxViTStage``: One MaxViT stage: optional strided MBConv downsample + N MaxViT blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.hybrid.maxvit.max_vi_t_block import MaxViTBlock
from hale_core.nn.vision.hybrid.maxvit.mb_conv import MBConv


class MaxViTStage(nn.Module):
    """One MaxViT stage: optional strided MBConv downsample + N MaxViT blocks.

    Args:
        in_ch:     Input channels.
        out_ch:    Output channels.
        depth:     Number of MaxViT blocks.
        num_heads: Attention heads.
        block_size: Local window size.
        drop_path_rates: Per-block stochastic depth rates.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        depth: int,
        num_heads: int,
        block_size: int = 8,
        drop_path_rates: list | None = None,
    ) -> None:
        super().__init__()
        rates = drop_path_rates or [0.0] * depth

        # Strided MBConv to halve resolution and adjust channels at stage entry
        self.downsample = MBConv(in_ch, out_ch, stride=2, drop_path=rates[0])

        # Remaining MaxViT blocks (all same channel count, stride=1)
        self.blocks = nn.ModuleList(
            [
                MaxViTBlock(out_ch, num_heads, block_size=block_size, drop_path=rates[i + 1])
                for i in range(depth - 1)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`MaxViTStage`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x
