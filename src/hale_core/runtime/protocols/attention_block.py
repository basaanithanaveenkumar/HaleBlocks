"""``AttentionBlock``.

Structural type of an attention module: ``forward(x, attn_mask, key_padding_mask)``.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class AttentionBlock(Protocol):
    """Structural type of an attention module: ``forward(x, attn_mask, key_padding_mask)``."""

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of :class:`AttentionBlock`.

        Args:
            x: input (``torch.Tensor``).
            attn_mask: input (``torch.Tensor | None``).
            key_padding_mask: input (``torch.Tensor | None``).

        Returns:
            ``torch.Tensor``.
        """
        ...
