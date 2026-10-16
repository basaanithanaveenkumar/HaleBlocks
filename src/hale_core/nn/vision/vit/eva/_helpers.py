"""Shared helpers of :mod:`hale_core.nn.vision.vit.eva` (EVA-02 Vision Transformer Backbone)."""

from __future__ import annotations

import torch


def _build_rope_freqs(
    grid_size: int,
    head_dim: int,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute RoPE sin/cos frequencies for a 2D grid.

    Each head has `head_dim` dimensions.  We use head_dim/2 for row positions
    and head_dim/2 for column positions.

    Returns:
        cos_freq: [N, head_dim]  (N = grid_size²)
        sin_freq: [N, head_dim]
    """
    # Frequency bands — one per pair of dimensions
    half = head_dim // 4  # row uses half, col uses half
    theta = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))

    # Grid positions
    gy = torch.arange(grid_size, dtype=torch.float32)
    gx = torch.arange(grid_size, dtype=torch.float32)
    # Outer product → N positions
    py = gy.unsqueeze(1).expand(grid_size, grid_size).reshape(-1)  # [N]
    px = gx.unsqueeze(0).expand(grid_size, grid_size).reshape(-1)  # [N]

    # Angles per position per frequency
    ang_y = py.unsqueeze(1) * theta.unsqueeze(0)  # [N, half]
    ang_x = px.unsqueeze(1) * theta.unsqueeze(0)  # [N, half]

    # Concat row and col angles → [N, head_dim/2]
    ang = torch.cat([ang_y, ang_x], dim=1)  # [N, head_dim/2]

    # Duplicate so each pair gets the same angle: [N, head_dim]
    # Pattern: [θ₀, θ₀, θ₁, θ₁, ...] → cos/sin applied to (x₀, x₁) pair
    ang = ang.repeat_interleave(2, dim=1)

    return ang.cos(), ang.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of consecutive dimensions: [a, b, c, d] → [-b, a, -d, c]."""
    x1, x2 = x[..., ::2], x[..., 1::2]  # split even/odd dims
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE rotation to Q and K tensors.

    Args:
        q, k:    [B, H, N, d]
        cos, sin: [N, d]  (pre-computed for the sequence length N)

    Returns:
        q', k' with RoPE applied — same shape.
    """
    # Broadcast cos/sin over batch and head dimensions
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, N, d]
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot
