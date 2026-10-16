"""``InceptionModuleV1``: Inception module as defined in GoogLeNet (Szegedy et al. 2015)."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.cnn.inception.conv_bn_relu import ConvBnRelu


class InceptionModuleV1(nn.Module):
    """Inception module as defined in GoogLeNet (Szegedy et al. 2015).

    Four parallel branches computed independently then concatenated on the
    channel dimension, capturing features at multiple receptive field sizes.

    Args:
        in_ch: Input channels.
        c1:    Branch-1 output channels (1×1 conv).
        c2_r:  Branch-2 reduction (1×1 before 3×3).
        c2:    Branch-2 output (3×3 conv).
        c3_r:  Branch-3 reduction (1×1 before 5×5).
        c3:    Branch-3 output (5×5 conv).
        c4:    Branch-4 output (1×1 after MaxPool).
    """

    def __init__(
        self,
        in_ch: int,
        c1: int,
        c2_r: int,
        c2: int,
        c3_r: int,
        c3: int,
        c4: int,
    ) -> None:
        super().__init__()
        self.branch1 = ConvBnRelu(in_ch, c1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBnRelu(in_ch, c2_r, kernel_size=1),
            ConvBnRelu(c2_r, c2, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvBnRelu(in_ch, c3_r, kernel_size=1),
            ConvBnRelu(c3_r, c3, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBnRelu(in_ch, c4, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1
        )
