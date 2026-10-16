"""``_InceptionTypeB``: Inception-v3 Type-B: factorised n×n → 1×n + n×1 (asymmetric)."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.cnn.inception.conv_bn_relu import ConvBnRelu


class _InceptionTypeB(nn.Module):
    """Inception-v3 Type-B: factorised n×n → 1×n + n×1 (asymmetric)."""

    def __init__(self, in_ch: int, channels_7x7: int) -> None:
        super().__init__()
        c7 = channels_7x7
        self.br1 = ConvBnRelu(in_ch, 192, kernel_size=1)
        self.br2 = nn.Sequential(
            ConvBnRelu(in_ch, c7, kernel_size=1),
            ConvBnRelu(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
            ConvBnRelu(c7, 192, kernel_size=(7, 1), padding=(3, 0)),
        )
        self.br3 = nn.Sequential(
            ConvBnRelu(in_ch, c7, kernel_size=1),
            ConvBnRelu(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            ConvBnRelu(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
            ConvBnRelu(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            ConvBnRelu(c7, 192, kernel_size=(1, 7), padding=(0, 3)),
        )
        self.br4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBnRelu(in_ch, 192, kernel_size=1),
        )
        # Total: 192 * 4 = 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.br1(x), self.br2(x), self.br3(x), self.br4(x)], dim=1)
