"""``_InceptionTypeA``: Inception-v3 Type-A: replaces 5×5 with two 3×3 convs."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.cnn.inception.conv_bn_relu import ConvBnRelu


class _InceptionTypeA(nn.Module):
    """Inception-v3 Type-A: replaces 5×5 with two 3×3 convs."""

    def __init__(self, in_ch: int, pool_features: int) -> None:
        super().__init__()
        self.br1 = ConvBnRelu(in_ch, 64, kernel_size=1)
        self.br2 = nn.Sequential(
            ConvBnRelu(in_ch, 48, kernel_size=1),
            ConvBnRelu(48, 64, kernel_size=5, padding=2),
        )
        self.br3 = nn.Sequential(
            ConvBnRelu(in_ch, 64, kernel_size=1),
            ConvBnRelu(64, 96, kernel_size=3, padding=1),
            ConvBnRelu(96, 96, kernel_size=3, padding=1),
        )
        self.br4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBnRelu(in_ch, pool_features, kernel_size=1),
        )
        # Total: 64+64+96+pool_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`_InceptionTypeA`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        return torch.cat([self.br1(x), self.br2(x), self.br3(x), self.br4(x)], dim=1)
