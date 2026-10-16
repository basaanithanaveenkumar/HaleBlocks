"""``DeiT3Attention``: Standard multi-head self-attention for DeiT III."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.vit.vit import _trunc_normal_


class DeiT3Attention(nn.Module):
    """Standard multi-head self-attention for DeiT III.

    Uses PyTorch 2.0 fused kernel (F.scaled_dot_product_attention) when available.
    No register tokens, no LayerScale inside the attention — LayerScale is on the block.
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
        self._use_flash = hasattr(F, "scaled_dot_product_attention")

        self.in_proj = nn.Linear(dim, dim * 3, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

        _trunc_normal_(self.in_proj.weight)
        _trunc_normal_(self.out_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = x.shape
        H = self.num_heads

        qkv = self.in_proj(x).reshape(B, N, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self._use_flash and not return_attn:
            drop_p = self.attn_drop_p if self.training else 0.0
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)
            attn_weights = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            if self.training and self.attn_drop_p > 0:
                attn = F.dropout(attn, p=self.attn_drop_p)
            attn_weights = attn if return_attn else None
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.out_proj(x)), attn_weights
