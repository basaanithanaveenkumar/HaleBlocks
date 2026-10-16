"""``StochasticDepth``: Stochastic depth regularisation (Huang et al. 2016)."""

from __future__ import annotations

import torch
import torch.nn as nn


class StochasticDepth(nn.Module):
    """Stochastic depth regularisation (Huang et al. 2016).

    Randomly drops the entire forward path of a block during training,
    equivalent to randomly shortening the effective depth of the network.
    At inference, acts as identity scaled by survival probability.

    Args:
        drop_prob: Probability of dropping (zeroing) the block output.
    """

    def __init__(self, drop_prob: float) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        # Random binary mask: shape [B, 1, 1, 1] — drops entire samples in the batch
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.rand(shape, dtype=x.dtype, device=x.device)
        noise = torch.floor(noise + keep_prob)
        # Scale by 1/keep_prob to maintain expected value (like dropout)
        return x * noise / keep_prob
