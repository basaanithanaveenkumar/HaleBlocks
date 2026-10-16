"""``SqueezeExcite``: Channel attention: avgpool → FC(squeeze) → ReLU → FC(excite) → sigmoid."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcite(nn.Module):
    """Channel attention: avgpool → FC(squeeze) → ReLU → FC(excite) → sigmoid."""

    def __init__(self, in_channels: int, reduction_ratio: int = 4) -> None:
        super().__init__()
        mid = max(1, in_channels // reduction_ratio)
        self.fc1 = nn.Linear(in_channels, mid)
        self.fc2 = nn.Linear(mid, in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=[2, 3])  # [B, C]
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale.unsqueeze(-1).unsqueeze(-1)
