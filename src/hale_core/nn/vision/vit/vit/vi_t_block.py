"""``ViTBlock``: Single ViT encoder block: pre-norm → attention → pre-norm → MLP."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.vit.vit.vi_t_attention import ViTAttention
from hale_core.nn.vision.vit.vit.vi_tmlp import ViTMLP


class ViTBlock(nn.Module):
    """Single ViT encoder block: pre-norm → attention → pre-norm → MLP.

    Pre-norm (LN before the residual branch) is more stable than post-norm,
    especially for large models and long sequences.

    Residual formulation:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
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
        self.attn = ViTAttention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=mlp_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = ViTMLP(dim, mlp_ratio=mlp_ratio, drop=mlp_drop)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of :class:`ViTBlock`.

        Args:
            x: input (``torch.Tensor``).
            return_attn: input (``bool``).

        Returns:
            ``tuple[torch.Tensor, Optional[torch.Tensor]]``.
        """
        attn_out, attn_weights = self.attn(self.norm1(x), return_attn=return_attn)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights
