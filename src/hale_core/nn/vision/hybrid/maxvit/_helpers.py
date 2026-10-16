"""Shared helpers of :mod:`hale_core.nn.vision.hybrid.maxvit` (MaxViT — Multi-Axis Vision Transformer (Tu et al., 2022))."""

from __future__ import annotations

import torch


def _trunc_normal_(t: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    with torch.no_grad():
        return t.normal_(0, std).clamp_(-2 * std, 2 * std)


def block_partition(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Partition spatial map into non-overlapping blocks.

    Each block of P×P pixels is one "window" for local attention.

    Args:
        x:          [B, H, W, C]  (channels-last)
        block_size: P — local window height == width

    Returns:
        blocks: [B * nH * nW, P, P, C]  where nH=H//P, nW=W//P
    """
    B, H, W, C = x.shape
    nH, nW = H // block_size, W // block_size
    x = x.view(B, nH, block_size, nW, block_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B * nH * nW, block_size, block_size, C)


def block_reverse(blocks: torch.Tensor, block_size: int, H: int, W: int) -> torch.Tensor:
    """Reconstruct [B, H, W, C] from [B*nH*nW, P, P, C]."""
    nH, nW = H // block_size, W // block_size
    B = blocks.shape[0] // (nH * nW)
    x = blocks.view(B, nH, nW, block_size, block_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


def grid_partition(x: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Partition into dilated grids for global attention.

    Instead of contiguous patches, takes every grid_size-th pixel.
    This gives each grid window a full-image receptive field at O(G²) cost.

    Example: 32×32 image, grid_size=8 → 8×8=64 windows, each 4×4 tokens.

    Args:
        x:         [B, H, W, C]
        grid_size: G — stride between picked pixels

    Returns:
        grids: [B * G * G, H//G, W//G, C]
    """
    B, H, W, C = x.shape
    gh, gw = H // grid_size, W // grid_size
    x = x.view(B, gh, grid_size, gw, grid_size, C)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.view(B * grid_size * grid_size, gh, gw, C)


def grid_reverse(grids: torch.Tensor, grid_size: int, H: int, W: int) -> torch.Tensor:
    """Reconstruct [B, H, W, C] from [B*G*G, H//G, W//G, C]."""
    gh, gw = H // grid_size, W // grid_size
    B = grids.shape[0] // (grid_size * grid_size)
    x = grids.view(B, grid_size, grid_size, gh, gw, -1)
    return x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, H, W, -1)
