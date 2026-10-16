"""``LogContinuousPositionBias``: Log-spaced Continuous Position Bias (Log-CPB) for SwinV2."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LogContinuousPositionBias(nn.Module):
    """Log-spaced Continuous Position Bias (Log-CPB) for SwinV2.

    Replaces the fixed learnable table of SwinV1 with a small MLP that inputs
    log-compressed coordinate differences. This allows inference at resolutions
    different from training (the MLP generalises; a table does not).

    The MLP maps (2,) → (num_heads,) via two linear layers.
    """

    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        # Small MLP: 2 → 512 → num_heads (no bias on last layer for stability)
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )

    def _compute_bias(self, window_size: int) -> torch.Tensor:
        """Compute relative position bias for a given window size.

        Returns:
            bias: [num_heads, window_size*window_size, window_size*window_size]
        """
        coords_h = torch.arange(window_size, device=next(self.parameters()).device)
        coords_w = torch.arange(window_size, device=next(self.parameters()).device)
        # All pairwise (row, col) differences
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # [2, win, win]
        coords_flat = coords.flatten(1)  # [2, win*win]
        # Relative distances: [2, win*win, win*win]
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0).float()  # [win*win, win*win, 2]

        # Log-compress: sign(d) * log(1 + |d|)  — values in ≈ [-3.2, 3.2] for win=8
        rel_log = rel.sign() * torch.log1p(rel.abs())

        # Normalise to [-8, 8] so MLP sees consistent scale regardless of window_size
        norm_factor = math.log(window_size)
        rel_log = rel_log / norm_factor * 8.0

        # MLP → [win*win, win*win, num_heads]
        bias = self.cpb_mlp(rel_log)
        # Apply sigmoid to keep magnitudes bounded → scale to [-log(max_dist), log(max_dist)]
        bias = 16.0 * bias.sigmoid()
        # Rearrange to [num_heads, win*win, win*win]
        bias = bias.permute(2, 0, 1).contiguous()
        return bias

    def forward(self, window_size: int) -> torch.Tensor:
        """Return the relative position bias for ``window_size``, recomputed on every call.

        The bias must not be cached: it is a function of the trainable ``cpb_mlp``. A cached
        tensor either has no autograd graph (if first computed under ``no_grad`` / in eval, so
        the MLP never trains) or a graph that is freed after the first ``backward()`` (so the
        second training step fails), and it goes stale after every optimiser step or
        ``load_state_dict``. The MLP runs on only window² × window² coordinate pairs.
        """
        return self._compute_bias(window_size)
