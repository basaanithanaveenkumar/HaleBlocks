"""``_InceptionTypeC``: Inception-v3 Type-C: splits 3×3 into 1×3 + 3×1 in parallel (wide)."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.cnn.inception.conv_bn_relu import ConvBnRelu


class _InceptionTypeC(nn.Module):
    """Inception-v3 Type-C: splits 3×3 into 1×3 + 3×1 in parallel (wide).
    Used in deepest stage for maximum filter bank diversity."""

    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.br1 = ConvBnRelu(in_ch, 320, kernel_size=1)
        self.br2_r = ConvBnRelu(in_ch, 384, kernel_size=1)
        self.br2_1x3 = ConvBnRelu(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.br2_3x1 = ConvBnRelu(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.br3_r = ConvBnRelu(in_ch, 448, kernel_size=1)
        self.br3_3x3 = ConvBnRelu(448, 384, kernel_size=3, padding=1)
        self.br3_1x3 = ConvBnRelu(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.br3_3x1 = ConvBnRelu(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.br4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBnRelu(in_ch, 192, kernel_size=1),
        )
        # Total: 320 + 384+384 + 384+384 + 192 = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`_InceptionTypeC`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        b1 = self.br1(x)
        b2_r = self.br2_r(x)
        b2 = torch.cat([self.br2_1x3(b2_r), self.br2_3x1(b2_r)], dim=1)  # 768
        b3_r = self.br3_3x3(self.br3_r(x))
        b3 = torch.cat([self.br3_1x3(b3_r), self.br3_3x1(b3_r)], dim=1)  # 768
        b4 = self.br4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)  # 320+768+768+192 = 2048
