"""Shared helpers of :mod:`hale_core.nn.vision.swin.swinv2` (SwinV2 — Swin Transformer V2 (Liu et
al.,
2022)).
"""

from __future__ import annotations

import torch


def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    """Truncated normal initialization (values clipped to ±2σ)."""
    with torch.no_grad():
        return tensor.normal_(0, std).clamp_(-2 * std, 2 * std)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition spatial feature map into non-overlapping windows.

    Args:
        x:           [B, H, W, C]
        window_size: Window height == width

    Returns:
        windows: [num_windows*B, window_size, window_size, C]
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # Reorder to [B, nH, nW, win, win, C] → [B*nH*nW, win, win, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, C)


def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """Reconstruct feature map from windows.

    Args:
        windows:     [num_windows*B, window_size, window_size, C]
        window_size: Window height == width
        H, W:        Original spatial dimensions

    Returns:
        x: [B, H, W, C]
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x
