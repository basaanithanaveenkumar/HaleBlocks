"""``DropPath``: Stochastic depth — drop entire residual branch per sample."""

from __future__ import annotations

import torch
import torch.nn as nn


class DropPath(nn.Module):
    """Stochastic depth — drop entire residual branch per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`DropPath`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        if not self.training or self.drop_prob == 0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.rand(shape, device=x.device, dtype=x.dtype).floor_() + keep
        return x.div(keep) * noise
