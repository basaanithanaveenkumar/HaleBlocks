"""``GRN``: Global Response Normalization (GRN) — the key addition in ConvNeXt V2."""

from __future__ import annotations

import torch
import torch.nn as nn


class GRN(nn.Module):
    """Global Response Normalization (GRN) — the key addition in ConvNeXt V2.

    GRN introduces cross-channel feature competition: channels with large global
    norms suppress channels with small global norms.  This discourages feature
    collapse (redundant channels learning the same thing) during masked autoencoding.

    Formulation (Woo et al. 2023, Eq. 1–3):
        gx = ‖x‖₂ over H, W dims          → [B, 1, 1, C]  (global spatial norm)
        nx = gx / E[gx]                    → normalised norms (E over C)
        output = x * nx + gamma * x + beta
    where gamma and beta are learned per-channel parameters (shape [C]).

    Args:
        dim: Number of feature channels (C).

    Shape:
        Input / Output: [B, H, W, C]  (channels-last convention inside blocks)
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        # Learnable scale and bias initialised to 0 (start as identity)
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Global spatial L2 norm per channel: [B, 1, 1, C]
        gx = x.norm(p=2, dim=(1, 2), keepdim=True)
        # Normalise across the channel dimension
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        # Scale and shift
        return self.gamma * (x * nx) + self.beta + x
