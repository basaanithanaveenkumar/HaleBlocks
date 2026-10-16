"""Shared helpers of :mod:`hale_core.nn.vision.cnn.nfnet` (NFNet Vision Backbones)."""

from __future__ import annotations

import torch
import torch.nn as nn


def agc_clip(model: nn.Module, lambda_: float = 0.01) -> None:
    """Adaptive Gradient Clipping: clips each parameter's gradient by its weight norm.

    Per-parameter clipping: g ← g * min(1, λ * ||w|| / ||g||).
    Should be called after loss.backward() before optimizer.step().

    Args:
        model:   The NFNet model (or any nn.Module with gradients).
        lambda_: Clipping coefficient (default 0.01, as in the paper).
    """
    for p in model.parameters():
        if p.grad is None:
            continue
        # Unit-wise norms (paper Sec. 4.1): one norm per output unit = per row of the weight,
        # i.e. over every dim except dim 0. Vectors (biases, gains) are clipped element-wise.
        # torch.linalg.vector_norm is required: Tensor.norm(dim=[1, 2, 3]) uses the Frobenius
        # *matrix* norm, which rejects more than two dims on recent PyTorch.
        dims = tuple(range(1, p.ndim))
        if dims:
            param_norm = torch.linalg.vector_norm(p.detach(), dim=dims, keepdim=True)
            grad_norm = torch.linalg.vector_norm(p.grad, dim=dims, keepdim=True)
        else:
            param_norm, grad_norm = p.detach().abs(), p.grad.abs()
        param_norm = param_norm.clamp(min=1e-3)
        max_norm = param_norm * lambda_
        clip_coef = (max_norm / grad_norm.clamp(min=1e-6)).clamp(max=1.0)
        p.grad.mul_(clip_coef)


_NFNET_WIDTHS = [256, 512, 1536, 1536]


_NFNET_DEPTHS = {
    "f0": [1, 2, 6, 3],
    "f1": [2, 4, 12, 6],
    "f2": [3, 6, 18, 9],
    "f3": [4, 8, 24, 12],
    "f4": [5, 10, 30, 15],
    "f5": [6, 12, 36, 18],
}
