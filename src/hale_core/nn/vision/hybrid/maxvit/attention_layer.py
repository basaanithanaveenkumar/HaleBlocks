"""``AttentionLayer``: Pre-LN attention + FFN sub-layer used in MaxViT blocks."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.hybrid.maxvit.drop_path import DropPath
from hale_core.nn.vision.hybrid.maxvit.window_attention import WindowAttention


class AttentionLayer(nn.Module):
    """Pre-LN attention + FFN sub-layer used in MaxViT blocks.

    Layout:
      x → LN → attn(x) → + residual → LN → FFN → + residual

    Args:
        dim:         Channel dimension.
        num_heads:   Attention heads.
        window_size: Attention window size.
        mlp_ratio:   FFN expand ratio.
        drop_path:   Stochastic depth rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)

        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B_win, N, C]"""
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x
