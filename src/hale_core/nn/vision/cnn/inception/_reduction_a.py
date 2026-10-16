"""``_ReductionA``: Grid reduction A: 288 → 768 channels with stride 2."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.cnn.inception.conv_bn_relu import ConvBnRelu


class _ReductionA(nn.Module):
    """Grid reduction A: 288 → 768 channels with stride 2."""

    def __init__(self, in_ch: int = 288) -> None:
        super().__init__()
        # Branch 1: 3×3 stride-2 conv
        self.br1 = ConvBnRelu(in_ch, 384, kernel_size=3, stride=2)
        # Branch 2: 1×1 → 3×3 → 3×3 stride-2
        self.br2 = nn.Sequential(
            ConvBnRelu(in_ch, 64, kernel_size=1),
            ConvBnRelu(64, 96, kernel_size=3, padding=1),
            ConvBnRelu(96, 96, kernel_size=3, stride=2),
        )
        # Branch 3: MaxPool stride-2 (keeps in_ch channels)
        self.br3 = nn.MaxPool2d(kernel_size=3, stride=2)
        # Total: 384 + 96 + 288 = 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.br1(x), self.br2(x), self.br3(x)], dim=1)
