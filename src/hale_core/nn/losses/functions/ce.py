"""Plain cross-entropy token loss (registered as ``ce``)."""

from __future__ import annotations

import torch

from hale_core.nn.losses.reduce import per_token_cross_entropy
from hale_core.nn.losses.registry import register_token_loss


@register_token_loss("ce")
def cross_entropy_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
    **_,
) -> torch.Tensor:
    """Per-token cross-entropy; positions equal to ``ignore_index`` get zero loss."""
    return per_token_cross_entropy(logits, targets, ignore_index=ignore_index)
