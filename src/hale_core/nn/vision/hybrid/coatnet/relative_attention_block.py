"""``RelativeAttentionBlock``: Transformer block with relative position bias (used in S3, S4)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.hybrid.coatnet.drop_path import DropPath
from hale_core.nn.vision.hybrid.coatnet.relative_position_bias import RelativePositionBias


class RelativeAttentionBlock(nn.Module):
    """Transformer block with relative position bias (used in S3, S4).

    Layout (pre-norm):
      x → LN → MHSA → + residual → LN → FFN → + residual

    Relative position bias is added to QK^T before softmax.

    Args:
        dim:       Channel dimension.
        num_heads: Attention heads.
        mlp_ratio: FFN hidden expand ratio.
        drop_path: Stochastic depth rate.
        max_dist:  Maximum relative position distance for bias table.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        max_dist: int = 64,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

        self.rel_bias = RelativePositionBias(num_heads, max_dist=max_dist)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def _attn(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Full self-attention with relative position bias over H×W spatial grid."""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # [B, heads, N, d]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Add relative position bias [heads, N, N]
        attn = attn + self.rel_bias(H, W).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: [B, H*W, C]  — flattened spatial tokens
        Returns:
            x: [B, H*W, C]
        """
        x = x + self.drop_path(self._attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x
