"""``FeedForwardBlock``: Structural type of a feed-forward module: ``forward(x) -> x``."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class FeedForwardBlock(Protocol):
    """Structural type of a feed-forward module: ``forward(x) -> x``."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block to ``x`` (same shape in and out)."""
        ...
