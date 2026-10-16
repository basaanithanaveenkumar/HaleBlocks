"""``InternImageFFN``: Feed-Forward Network: Linear → GELU → Dropout → Linear → Dropout."""

from __future__ import annotations

import torch
import torch.nn as nn


class InternImageFFN(nn.Module):
    """Feed-Forward Network: Linear → GELU → Dropout → Linear → Dropout."""

    def __init__(self, channels: int, expand: int = 4, drop: float = 0.0) -> None:
        super().__init__()
        mid = channels * expand
        self.fc1 = nn.Linear(channels, mid)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(mid, channels)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))
