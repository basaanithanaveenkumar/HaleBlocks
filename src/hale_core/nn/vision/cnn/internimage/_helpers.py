"""Shared helpers of :mod:`hale_core.nn.vision.cnn.internimage` (InternImage — DCNv3-based Vision
Backbone (Wang et al. 2022)).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _DropPath(nn.Module):
    """Stochastic depth: drop entire residual path for a sample."""

    def __init__(self, drop_prob: float) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`_DropPath`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        if not self.training or self.drop_prob == 0.0:
            return x
        survival = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.empty(shape, device=x.device, dtype=x.dtype).bernoulli_(survival)
        return x * noise.div_(survival)
