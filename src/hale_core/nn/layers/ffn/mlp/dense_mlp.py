"""``DenseMLP``: Two-layer feed-forward block (Linear → GELU → dropout → Linear)."""

from __future__ import annotations

import torch.nn as nn


class DenseMLP(nn.Module):
    """Two-layer feed-forward block (Linear → GELU → dropout → Linear)."""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        """Forward pass of :class:`DenseMLP`.

        Args:
            x: input.
        """
        return self.net(x)
