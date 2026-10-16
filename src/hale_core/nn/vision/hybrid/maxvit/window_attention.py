"""``WindowAttention``: Vanilla multi-head self-attention with 2D relative position bias."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.hybrid.maxvit._helpers import _trunc_normal_


class WindowAttention(nn.Module):
    """Vanilla multi-head self-attention with 2D relative position bias.

    Used for both block and grid attention in MaxViT.
    The relative bias table is indexed by (row_diff, col_diff) pairs,
    similar to Swin but applied to a small fixed-size window.

    Args:
        dim:         Channel dimension.
        num_heads:   Attention heads.
        window_size: Spatial size of the attention window (H == W assumed).
        attn_drop:   Attention weight dropout.
        proj_drop:   Output projection dropout.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Relative position bias: table [2*win-1, 2*win-1, num_heads]
        table_size = (2 * window_size - 1) ** 2
        self.rel_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))
        _trunc_normal_(self.rel_bias_table, std=0.02)

        # Precompute relative position index for a window_size × window_size grid
        coords = torch.stack(
            torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij")
        ).flatten(1)  # [2, win*win]
        rel = coords[:, :, None] - coords[:, None, :]  # [2, N, N]
        rel = rel.permute(1, 2, 0)  # [N, N, 2]
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("rel_pos_index", rel.sum(-1))  # [N, N]

    def _build_rel_index(self, win: int, device: torch.device) -> torch.Tensor:
        """Compute relative position index for a win×win grid.

        Returns:
            index: [win*win, win*win] — each entry is a row-index into rel_bias_table
        """
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(win, device=device),
                torch.arange(win, device=device),
                indexing="ij",
            )
        ).flatten(1)  # [2, win*win]
        rel = coords[:, :, None] - coords[:, None, :]  # [2, N, N]
        rel = rel.permute(1, 2, 0)  # [N, N, 2]
        rel[:, :, 0] += win - 1
        rel[:, :, 1] += win - 1
        rel[:, :, 0] *= 2 * win - 1
        return rel.sum(-1)  # [N, N]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B_win, N, C]  where N = window_size * window_size (may vary for grid attn)"""
        Bw, N, C = x.shape
        qkv = self.qkv(x).reshape(Bw, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Relative position bias — recompute index dynamically when actual window
        # size differs from construction-time window_size (e.g. grid attention).
        win = int(N**0.5)
        if win * win == N and win == self.window_size:
            idx = self.rel_pos_index
        else:
            idx = self._build_rel_index(win, x.device)

        # Clamp indices so they stay in-bounds if the table was sized for a
        # larger window; out-of-range entries get the nearest valid bias.
        idx_clamped = idx.view(-1).clamp(0, self.rel_bias_table.shape[0] - 1)
        bias = self.rel_bias_table[idx_clamped].view(N, N, self.num_heads).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(Bw, N, C)
        return self.proj_drop(self.proj(out))
