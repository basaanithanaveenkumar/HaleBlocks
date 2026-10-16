"""``FeedForwardBlock``: FeedForwardBlock (Structural types so training/eval depend on interfaces, not concrete variants)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class FeedForwardBlock(Protocol):
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
