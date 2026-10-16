"""``StochasticDepth``: Drop entire residual branches with probability `drop_prob`."""

from __future__ import annotations

import torch
import torch.nn as nn


class StochasticDepth(nn.Module):
    """Drop entire residual branches with probability `drop_prob`.

    At training time, each sample in a batch has probability `drop_prob` of
    having its residual branch zeroed out (the skip connection always remains).
    At test time, the branch is always active (scaled by survival_prob).

    This is the "per-sample" stochastic depth from Huang et al. 2016.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        survival = 1.0 - self.drop_prob
        # Shape [B, 1, 1, 1] for broadcasting over spatial dims
        noise = torch.empty(x.shape[0], 1, 1, 1, device=x.device, dtype=x.dtype)
        noise = noise.bernoulli_(survival)
        if survival > 0.0:
            noise.div_(survival)  # scale to maintain expected value
        return x * noise
