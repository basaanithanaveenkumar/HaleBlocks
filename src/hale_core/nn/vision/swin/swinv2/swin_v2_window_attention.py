"""``SwinV2WindowAttention``: Scaled cosine self-attention within a single window."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.swin.swinv2.log_continuous_position_bias import LogContinuousPositionBias


class SwinV2WindowAttention(nn.Module):
    """Scaled cosine self-attention within a single window.

    SwinV2 change: replaces dot-product Q·K^T with cosine similarity / tau
    to prevent logit explosion in large models.

    Args:
        dim:         Input channel count.
        window_size: Window height == width.
        num_heads:   Number of attention heads.
        attn_drop:   Dropout on attention weights.
        proj_drop:   Dropout on output projection.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        # Q, K, V projection (all-in-one for efficiency)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Log-CPB for resolution-agnostic relative position bias
        self.rel_pos_bias = LogContinuousPositionBias(num_heads)

        # Learnable per-head log-scale temperature (init=log(10) ≈ 2.30)
        # tau = exp(logit_scale.clip(max=log(100))) to prevent collapse
        self.logit_scale = nn.Parameter(torch.full((num_heads, 1, 1), math.log(10.0)))

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x:    [B*num_windows, win*win, C]
            mask: [num_windows, win*win, win*win] or None
        Returns:
            out: [B*num_windows, win*win, C]
        """
        Bw, N, C = x.shape
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(Bw, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, Bw, heads, N, head_dim]
        q, k, v = qkv.unbind(0)

        # Normalise Q and K to unit length for cosine similarity
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Temperature-scaled cosine attention
        tau = torch.clamp(self.logit_scale, max=math.log(100.0)).exp()
        attn = (q @ k.transpose(-2, -1)) * tau  # [Bw, heads, N, N]

        # Add Log-CPB relative position bias
        rpb = self.rel_pos_bias(self.window_size)  # [heads, N, N]
        attn = attn + rpb.unsqueeze(0)

        # Shifted-window attention mask: additive −∞ for cross-region pairs
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(Bw // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(Bw, N, C)
        return self.proj_drop(self.proj(out))
