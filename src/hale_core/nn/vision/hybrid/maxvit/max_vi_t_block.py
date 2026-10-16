"""``MaxViTBlock``: Single MaxViT block: MBConv → Block Attention → Grid Attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.hybrid.maxvit._helpers import (
    block_partition,
    block_reverse,
    grid_partition,
    grid_reverse,
)
from hale_core.nn.vision.hybrid.maxvit.attention_layer import AttentionLayer
from hale_core.nn.vision.hybrid.maxvit.mb_conv import MBConv


class MaxViTBlock(nn.Module):
    """Single MaxViT block: MBConv → Block Attention → Grid Attention.

    The three sub-layers give each block:
      - MBConv: strong local conv features (inductive bias)
      - Block Attention: fine-grained local self-attention
      - Grid Attention: global dilated self-attention

    Args:
        dim:          Channel dimension.
        num_heads:    Attention heads for both attention layers.
        block_size:   Local window size for block attention (default 8).
        mlp_ratio:    FFN expand ratio.
        drop_path:    Stochastic depth rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        block_size: int = 8,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.block_size = block_size

        # MBConv for local spatial features (operates in channels-first)
        self.mbconv = MBConv(dim, dim, drop_path=drop_path)

        # Block (local window) attention
        self.block_attn = AttentionLayer(
            dim, num_heads, block_size, mlp_ratio=mlp_ratio, drop_path=drop_path
        )

        # Grid (global dilated) attention  — same window_size=block_size
        # because grid_partition creates windows of size (H//block_size) × (W//block_size)
        # but we use block_size as the grid stride, producing block_size² windows
        self.grid_attn = AttentionLayer(
            dim, num_heads, block_size, mlp_ratio=mlp_ratio, drop_path=drop_path
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]  (channels-first)
        Returns:
            x: [B, C, H, W]
        """
        # ── MBConv (channels-first) ───────────────────────────────────────────
        x = self.mbconv(x)

        B, C, H, W = x.shape
        P = self.block_size

        # Convert to channels-last for attention operations
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]

        # Pad to a multiple of block_size so partition math always divides evenly.
        # Both block attention (P×P windows) and grid attention (stride-P sampling)
        # require H and W to be multiples of P.  The pad is removed at the end.
        pad_h = (P - H % P) % P
        pad_w = (P - W % P) % P
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        # ── Block (local) Attention ───────────────────────────────────────────
        blocks = block_partition(x, P)  # [B*nH*nW, P, P, C]
        N = P * P
        blocks = blocks.view(-1, N, C)
        blocks = self.block_attn(blocks)
        blocks = blocks.view(-1, P, P, C)
        x = block_reverse(blocks, P, Hp, Wp)  # [B, Hp, Wp, C]

        # ── Grid (global) Attention ───────────────────────────────────────────
        # grid_partition samples every P-th location; each resulting window has
        # (Hp//P) × (Wp//P) tokens — size differs from P×P for block attention,
        # so WindowAttention uses a dynamic relative-bias index (see above).
        grids = grid_partition(x, P)  # [B*P*P, Hp//P, Wp//P, C]
        gh, gw = Hp // P, Wp // P
        grids = grids.view(-1, gh * gw, C)
        grids = self.grid_attn(grids)
        grids = grids.view(-1, gh, gw, C)
        x = grid_reverse(grids, P, Hp, Wp)  # [B, Hp, Wp, C]

        # Strip padding before returning
        if pad_h or pad_w:
            x = x[:, :H, :W, :].contiguous()

        # Back to channels-first
        return x.permute(0, 3, 1, 2).contiguous()
