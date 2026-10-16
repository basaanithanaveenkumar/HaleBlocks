"""``AttentionBlock``: AttentionBlock (Structural types so training/eval depend on interfaces, not concrete variants)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class AttentionBlock(Protocol):
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor: ...
