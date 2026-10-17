"""``ViTMLP``: Two-layer GELU MLP, expand ratio 4× (standard ViT FFN)."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.vit.vit._helpers import _trunc_normal_


class ViTMLP(nn.Module):
    """Two-layer GELU MLP, expand ratio 4× (standard ViT FFN).

    Structure: Linear(D → 4D) → GELU → Dropout → Linear(4D → D) → Dropout
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(drop)

        _trunc_normal_(self.fc1.weight)
        _trunc_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`ViTMLP`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        x = self.drop1(self.act(self.fc1(x)))
        return self.drop2(self.fc2(x))
