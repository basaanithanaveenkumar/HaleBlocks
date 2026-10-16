"""``RelativePositionBias``: Learnable relative position bias indexed by (row_diff, col_diff)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from hale_core.nn.vision.hybrid.coatnet._helpers import _trunc_normal_


class RelativePositionBias(nn.Module):
    """Learnable relative position bias indexed by (row_diff, col_diff).

    Table size: (2*max_dist-1) × (2*max_dist-1) per head.
    At each attention step, bias is indexed by the pair-wise position
    differences (clipped to [-max_dist+1, max_dist-1]).

    Args:
        num_heads: Number of attention heads.
        max_dist:  Maximum position difference to represent (default 64).
    """

    def __init__(self, num_heads: int, max_dist: int = 64) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.max_dist = max_dist
        table_size = (2 * max_dist - 1) ** 2
        self.table = nn.Parameter(torch.zeros(table_size, num_heads))
        _trunc_normal_(self.table, std=0.02)

        # Cache the index lookup for a given H×W grid
        self._cached_hw: Optional[tuple] = None
        self._cached_idx: Optional[torch.Tensor] = None

    def _build_index(self, H: int, W: int) -> torch.Tensor:
        """Precompute flat indices into the bias table for an H×W grid."""
        coords = torch.stack(
            torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        ).flatten(1)  # [2, H*W]
        # Pairwise differences
        rel_h = coords[0][:, None] - coords[0][None, :]  # [H*W, H*W]
        rel_w = coords[1][:, None] - coords[1][None, :]
        # Clip to [-max_dist+1, max_dist-1]
        md = self.max_dist - 1
        rel_h = rel_h.clamp(-md, md) + md
        rel_w = rel_w.clamp(-md, md) + md
        # Map to flat index into table of shape [2*max_dist-1, 2*max_dist-1]
        idx = rel_h * (2 * self.max_dist - 1) + rel_w  # [H*W, H*W]
        return idx

    def forward(self, H: int, W: int) -> torch.Tensor:
        """Return [num_heads, H*W, H*W] bias to add to attention logits."""
        if self._cached_hw != (H, W):
            self._cached_idx = self._build_index(H, W).to(self.table.device)
            self._cached_hw = (H, W)
        idx = self._cached_idx  # [H*W, H*W]
        N = H * W
        # Index table [table_size, num_heads] → [H*W, H*W, num_heads]
        bias = self.table[idx.view(-1)].view(N, N, self.num_heads)
        return bias.permute(2, 0, 1)  # [num_heads, H*W, H*W]
