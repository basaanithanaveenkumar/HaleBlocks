"""Shared helpers of :mod:`hale_core.nn.vision.hybrid.coatnet` (CoAtNet — Marrying Convolution and Attention (Dai et al., 2021))."""

from __future__ import annotations

import torch


def _trunc_normal_(t: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    with torch.no_grad():
        return t.normal_(0, std).clamp_(-2 * std, 2 * std)
