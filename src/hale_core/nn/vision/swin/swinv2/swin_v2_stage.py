"""``SwinV2Stage``: One stage: optional PatchMerging + N SwinV2 blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.swin.swinv2.patch_merging import PatchMerging
from hale_core.nn.vision.swin.swinv2.swin_v2_block import SwinV2Block


class SwinV2Stage(nn.Module):
    """One stage: optional PatchMerging + N SwinV2 blocks.

    Blocks alternate between regular (shift=False) and shifted (shift=True)
    window attention to capture both local and cross-window context.

    Args:
        dim:          Input channel count.
        depth:        Number of transformer blocks.
        num_heads:    Attention heads.
        window_size:  Window spatial size.
        downsample:   Whether to prepend a PatchMerging layer.
        drop_path_rates: List of stochastic-depth rates per block.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        downsample: bool = True,
        drop_path_rates: list | None = None,
    ) -> None:
        super().__init__()
        # Optional spatial downsampling at the beginning of each stage
        self.downsample = PatchMerging(dim // 2) if downsample else nn.Identity()
        # Output dim after optional merge
        out_dim = dim if downsample else dim

        rates = drop_path_rates or [0.0] * depth
        self.blocks = nn.ModuleList(
            [
                SwinV2Block(
                    dim=out_dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift=(i % 2 == 1),  # Alternate: even=regular, odd=shifted
                    drop_path=rates[i],
                )
                for i in range(depth)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, H, W, C] → [B, H', W', C'] (C' doubled if downsample)"""
        x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x
