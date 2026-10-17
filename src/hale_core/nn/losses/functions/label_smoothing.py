"""Label-smoothed cross-entropy token loss (registered as ``label_smoothing``)."""

from __future__ import annotations

import torch

from hale_core.nn.losses.reduce import per_token_cross_entropy
from hale_core.nn.losses.registry import register_token_loss


@register_token_loss("label_smoothing")
def label_smoothing_token_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.1,
    **_,
) -> torch.Tensor:
    """Per-token cross-entropy with ``label_smoothing`` mass spread over the vocabulary."""
    return per_token_cross_entropy(
        logits, targets, ignore_index=ignore_index, label_smoothing=label_smoothing
    )
