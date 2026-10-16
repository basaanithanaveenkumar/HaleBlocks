"""``_ReductionB``: Grid reduction B: 768 → 1280 channels with stride 2."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.cnn.inception.conv_bn_relu import ConvBnRelu


class _ReductionB(nn.Module):
    """Grid reduction B: 768 → 1280 channels with stride 2."""

    def __init__(self, in_ch: int = 768) -> None:
        super().__init__()
        self.br1 = nn.Sequential(
            ConvBnRelu(in_ch, 192, kernel_size=1),
            ConvBnRelu(192, 320, kernel_size=3, stride=2),
        )
        self.br2 = nn.Sequential(
            ConvBnRelu(in_ch, 192, kernel_size=1),
            ConvBnRelu(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            ConvBnRelu(192, 192, kernel_size=(7, 1), padding=(3, 0)),
            ConvBnRelu(192, 192, kernel_size=3, stride=2),
        )
        self.br3 = nn.MaxPool2d(kernel_size=3, stride=2)
        # Total: 320 + 192 + 768 = 1280

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.br1(x), self.br2(x), self.br3(x)], dim=1)
