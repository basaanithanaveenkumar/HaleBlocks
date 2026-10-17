"""``StochasticDepth``: Randomly drops the entire block output during training (Huang et al. 2016)."""

from __future__ import annotations

import torch
import torch.nn as nn


class StochasticDepth(nn.Module):
    """Randomly drops the entire block output during training (Huang et al. 2016)."""

    def __init__(self, drop_prob: float) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`StochasticDepth`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.floor(torch.rand(shape, dtype=x.dtype, device=x.device) + keep_prob)
        return x * noise / keep_prob
