"""``DropPath``: Stochastic depth: drop entire residual branch per-sample during training."""

from __future__ import annotations

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Stochastic depth: drop entire residual branch per-sample during training.

    At inference (eval mode) acts as identity.  The remaining branches are
    scaled up by 1/(1-p) so expected value is preserved (like Dropout).

    Args:
        drop_prob: Probability of dropping a branch in a given step.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.rand(shape, dtype=x.dtype, device=x.device).floor_() + keep_prob
        return x.div(keep_prob) * noise
