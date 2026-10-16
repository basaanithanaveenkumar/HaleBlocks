"""Shared helpers of :mod:`hale_core.nn.vision.vit.vit` (Vision Transformer (ViT) Backbone)."""

from __future__ import annotations

import torch
import torch.nn as nn


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    """Truncated normal init at ±2σ — standard for ViT weight init."""
    with torch.no_grad():
        # Box–Muller via normal + clamp, then rescale to correct distribution
        nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)
    return tensor
