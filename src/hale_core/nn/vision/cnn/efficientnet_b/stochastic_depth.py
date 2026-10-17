"""``StochasticDepth``: Randomly drops residual branches during training (Huang et al. 2016)."""

from __future__ import annotations

import torch
import torch.nn as nn


class StochasticDepth(nn.Module):
    """Randomly drops residual branches during training (Huang et al. 2016).

    At test time, the residual branch is always active (no scaling needed;
    the expected value is preserved by the Bernoulli / survival probability).
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
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
        survival = 1.0 - self.drop_prob
        noise = torch.empty(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype)
        noise.bernoulli_(survival).div_(survival)
        return x * noise
