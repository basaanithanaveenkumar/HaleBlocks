"""``CoAtNetAttnStage``: Sequence of Transformer blocks with optional spatial downsampling."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.hybrid.coatnet.relative_attention_block import RelativeAttentionBlock


class CoAtNetAttnStage(nn.Module):
    """Sequence of Transformer blocks with optional spatial downsampling.

    Downsampling: MaxPool2d(2) + 1×1 Conv before the transformer blocks.

    Args:
        in_ch:    Input channels.
        out_ch:   Output channels.
        depth:    Number of transformer blocks.
        num_heads: Attention heads.
        stride:   Spatial stride (2 = halve resolution with pool+conv).
        drop_path_rates: Per-block stochastic depth rates.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        depth: int,
        num_heads: int,
        stride: int = 2,
        drop_path_rates: list | None = None,
    ) -> None:
        super().__init__()
        rates = drop_path_rates or [0.0] * depth

        # Downsampling: pool for spatial halving, 1×1 conv for channel change
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = (
                nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
            )

        self.blocks = nn.ModuleList(
            [RelativeAttentionBlock(out_ch, num_heads, drop_path=rates[i]) for i in range(depth)]
        )
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] → [B, C', H', W']"""
        x = self.downsample(x)
        B, C, H, W = x.shape
        # Flatten spatial dims for transformer: [B, H*W, C]
        tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        for blk in self.blocks:
            tokens = blk(tokens, H, W)
        tokens = self.norm(tokens)
        # Restore spatial layout: [B, C, H, W]
        return tokens.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
