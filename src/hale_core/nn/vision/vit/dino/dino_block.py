"""``DINOBlock``: DINOv2 transformer block: pre-norm + LayerScale on both branches."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.vit.dino.dino_attention import DINOAttention
from hale_core.nn.vision.vit.dino.layer_scale import LayerScale

# Reuse PatchEmbed and MLP from sibling vit module (avoid duplication)
from hale_core.nn.vision.vit.vit import ViTMLP


class DINOBlock(nn.Module):
    """DINOv2 transformer block: pre-norm + LayerScale on both branches.

    Residual:
        x = x + LayerScale(Attention(LN(x)))
        x = x + LayerScale(MLP(LN(x)))
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        mlp_drop: float = 0.0,
        ls_init: float = 1e-5,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DINOAttention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=mlp_drop)
        self.ls1 = LayerScale(dim, init_value=ls_init)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = ViTMLP(dim, mlp_ratio=mlp_ratio, drop=mlp_drop)
        self.ls2 = LayerScale(dim, init_value=ls_init)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of :class:`DINOBlock`.

        Args:
            x: input (``torch.Tensor``).
            return_attn: input (``bool``).

        Returns:
            ``tuple[torch.Tensor, Optional[torch.Tensor]]``.
        """
        attn_out, attn_weights = self.attn(self.norm1(x), return_attn=return_attn)
        x = x + self.ls1(attn_out)
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x, attn_weights
