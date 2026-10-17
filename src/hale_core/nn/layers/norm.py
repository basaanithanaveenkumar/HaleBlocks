"""Normalisation layers — LayerNorm and RMSNorm — registered in NORMS_REGISTRY."""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Pre-norm RMSNorm (unit-initialised scale, no bias).

    Registered as ``"rms"`` in ``NORMS_REGISTRY``.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`RMSNorm`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).type_as(x) * self.weight


def _register_norms() -> None:
    from hale_core.registry.plugins import NORMS_REGISTRY

    if "layer" in NORMS_REGISTRY:
        return
    NORMS_REGISTRY.add("layer", nn.LayerNorm)
    NORMS_REGISTRY.add("rms", RMSNorm)


_register_norms()
