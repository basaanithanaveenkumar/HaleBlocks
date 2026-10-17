"""``ViTAttention``: Multi-head self-attention for Vision Transformer."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.vit.vit._helpers import _trunc_normal_


class ViTAttention(nn.Module):
    """Multi-head self-attention for Vision Transformer.

    Implements the standard scaled dot-product attention:
        Attn(Q,K,V) = softmax(QK^T / √d_head) · V

    Uses a single in_proj weight matrix [D, 3D] for combined Q/K/V — this is
    cache-friendly and matches the original Transformer paper's formulation.

    Args:
        dim:       Input/output dimensionality D.
        num_heads: Number of attention heads H. dim must be divisible by H.
        attn_drop: Dropout probability on attention weights (default 0.0).
        proj_drop: Dropout probability on the output projection (default 0.0).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5  # 1/√d_head for numerical stability

        # Combined Q, K, V projection — one matrix instead of three for efficiency
        self.in_proj = nn.Linear(dim, dim * 3, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        _trunc_normal_(self.in_proj.weight)
        _trunc_normal_(self.out_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Forward pass of :class:`ViTAttention`.

        Args:
            x: input (``torch.Tensor``).
            return_attn: input (``bool``).

        Returns:
            ``tuple[torch.Tensor, Optional[torch.Tensor]]``.
        """
        B, N, D = x.shape
        H = self.num_heads

        # Project and split into Q, K, V heads
        qkv = self.in_proj(x)  # [B, N, 3D]
        qkv = qkv.reshape(B, N, 3, H, self.head_dim)  # [B, N, 3, H, d]
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, N, d]
        q, k, v = qkv.unbind(0)  # each [B, H, N, d]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, H, N, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum of values
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        x = self.proj_drop(self.out_proj(x))

        return x, (attn if return_attn else None)
