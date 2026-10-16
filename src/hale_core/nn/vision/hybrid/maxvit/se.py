"""``SE``: Squeeze-Excite channel attention."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SE(nn.Module):
    """Squeeze-Excite channel attention."""

    def __init__(self, ch: int, rd: int = None) -> None:
        super().__init__()
        rd = rd or max(1, ch // 4)
        self.fc1 = nn.Linear(ch, rd)
        self.fc2 = nn.Linear(rd, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3))
        s = F.silu(self.fc1(s))
        s = self.fc2(s).sigmoid()
        return x * s[:, :, None, None]
