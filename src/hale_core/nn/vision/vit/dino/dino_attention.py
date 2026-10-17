"""``DINOAttention``: Multi-head self-attention with optional PyTorch 2.0 fused kernel."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse PatchEmbed and MLP from sibling vit module (avoid duplication)
from hale_core.nn.vision.vit.vit import _trunc_normal_


class DINOAttention(nn.Module):
    """Multi-head self-attention with optional PyTorch 2.0 fused kernel.

    When torch.nn.functional.scaled_dot_product_attention is available (PyTorch
    ≥ 2.0), uses the fused Flash Attention kernel for ~2× speedup on long
    sequences.  Falls back to manual attention on older versions.

    Args:
        dim:       Token dimension D.
        num_heads: Number of attention heads H.
        attn_drop: Dropout probability on attention weights (ignored for flash attn).
        proj_drop: Dropout probability on output projection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.attn_drop_p = attn_drop

        self.in_proj = nn.Linear(dim, dim * 3, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

        # Check once at init whether fused kernel is available
        self._use_flash = hasattr(F, "scaled_dot_product_attention")

        _trunc_normal_(self.in_proj.weight)
        _trunc_normal_(self.out_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of :class:`DINOAttention`.

        Args:
            x: input (``torch.Tensor``).
            return_attn: input (``bool``).

        Returns:
            ``tuple[torch.Tensor, Optional[torch.Tensor]]``.
        """
        B, N, D = x.shape
        H = self.num_heads

        qkv = self.in_proj(x).reshape(B, N, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each [B, H, N, d]

        if self._use_flash and not return_attn:
            # Fused Flash Attention — no explicit attention matrix materialised
            # dropout_p only applies during training; set 0.0 at eval
            drop_p = self.attn_drop_p if self.training else 0.0
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)
            attn_weights = None
        else:
            # Manual attention (needed when return_attn=True for visualisation)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            if self.training and self.attn_drop_p > 0:
                attn = F.dropout(attn, p=self.attn_drop_p)
            attn_weights = attn if return_attn else None
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.out_proj(x)), attn_weights
