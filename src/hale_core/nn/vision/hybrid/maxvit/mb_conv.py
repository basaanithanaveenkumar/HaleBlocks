"""``MBConv``: MBConv with BN pre-norm, depthwise conv and SE."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.hybrid.maxvit.drop_path import DropPath
from hale_core.nn.vision.hybrid.maxvit.se import SE


class MBConv(nn.Module):
    """MBConv with BN pre-norm, depthwise conv and SE.

    Used in MaxViT as the first sub-layer of each MaxViT block for
    local feature extraction before the attention layers.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expand: int = 4,
        stride: int = 1,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        mid = in_ch * expand
        self.use_res = (in_ch == out_ch) and (stride == 1)
        self.pre_norm = nn.BatchNorm2d(in_ch)

        self.expand = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.dw = nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False)
        self.dw_norm = nn.BatchNorm2d(mid)
        self.se = SE(mid)
        self.project = nn.Conv2d(mid, out_ch, 1, bias=False)
        self.proj_norm = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`MBConv`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        residual = x
        x = self.pre_norm(x)
        x = self.act(self.expand(x))
        x = self.act(self.dw_norm(self.dw(x)))
        x = self.se(x)
        x = self.proj_norm(self.project(x))
        if self.use_res:
            x = residual + self.drop_path(x)
        return x
