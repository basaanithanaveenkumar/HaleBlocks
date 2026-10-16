"""``EVAAttention``: Multi-head self-attention with RoPE and per-QK LayerNorm."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.vit.eva._helpers import _apply_rope
from hale_core.nn.vision.vit.vit import _trunc_normal_


class EVAAttention(nn.Module):
    """Multi-head self-attention with RoPE and per-QK LayerNorm.

    Sub-LN: Separate LayerNorm on Q and K before computing attention.
    This prevents attention logit explosion observed in very large models.

    RoPE: The pre-computed cos/sin buffers must be passed from the backbone
    (they depend on the image resolution, which is fixed at __init__ time).

    Args:
        dim:       Token dimension D.
        num_heads: Number of attention heads H.
        attn_drop: Attention weight dropout.
        proj_drop: Output projection dropout.
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

        # Sub-LN: LayerNorm on Q and K to stabilise deep network training
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

        _trunc_normal_(self.in_proj.weight)
        _trunc_normal_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = x.shape
        H = self.num_heads

        qkv = self.in_proj(x).reshape(B, N, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each [B, H, N, d]

        # Sub-LN on Q and K (per-head, over head_dim axis)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply 2D RoPE to Q and K — positions come from the spatial grid
        q, k = _apply_rope(q, k, rope_cos.to(q.dtype), rope_sin.to(k.dtype))

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        if self.training and self.attn_drop_p > 0:
            attn = F.dropout(attn, p=self.attn_drop_p)
        attn_weights = attn if return_attn else None

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.out_proj(x)), attn_weights
