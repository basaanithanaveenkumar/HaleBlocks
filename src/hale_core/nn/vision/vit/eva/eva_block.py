"""``EVABlock``: EVA-02 transformer block: pre-norm, RoPE attention, SwiGLU FFN."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from hale_core.nn.vision.vit.eva.eva_attention import EVAAttention
from hale_core.nn.vision.vit.eva.swi_gluffn import SwiGLUFFN


class EVABlock(nn.Module):
    """EVA-02 transformer block: pre-norm, RoPE attention, SwiGLU FFN.

    RoPE frequencies are passed from the backbone and not stored per-block
    to avoid duplicating large buffers in memory.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        mlp_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EVAAttention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=mlp_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwiGLUFFN(dim, mlp_ratio=mlp_ratio, drop=mlp_drop)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass of :class:`EVABlock`.

        Args:
            x: input (``torch.Tensor``).
            rope_cos: input (``torch.Tensor``).
            rope_sin: input (``torch.Tensor``).
            return_attn: input (``bool``).

        Returns:
            ``tuple[torch.Tensor, Optional[torch.Tensor]]``.
        """
        attn_out, attn_weights = self.attn(
            self.norm1(x), rope_cos, rope_sin, return_attn=return_attn
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights
