"""``InternImageBlock``: Single InternImage transformer block."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.cnn.internimage._helpers import _DropPath
from hale_core.nn.vision.cnn.internimage.dc_nv3 import DCNv3
from hale_core.nn.vision.cnn.internimage.intern_image_ffn import InternImageFFN


class InternImageBlock(nn.Module):
    """Single InternImage transformer block.

    Structure (pre-norm convention):
        x → LN → DCNv3 → x (residual)
        x → LN → FFN   → x (residual)

    Pre-norm stabilises training for large models; same design as ViT.

    Args:
        channels:    Feature channels.
        groups:      DCNv3 deformable groups.
        kernel_size: DCNv3 sampling grid size.
        ffn_expand:  FFN expansion ratio.
        drop:        Dropout for FFN.
        drop_path:   Stochastic depth probability.
    """

    def __init__(
        self,
        channels: int,
        groups: int,
        kernel_size: int = 3,
        ffn_expand: int = 4,
        drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(channels, eps=1e-6)
        self.dcnv3 = DCNv3(channels, kernel_size=kernel_size, groups=groups)
        self.norm2 = nn.LayerNorm(channels, eps=1e-6)
        self.ffn = InternImageFFN(channels, ffn_expand, drop)
        # Stochastic depth on both branches
        self.drop_path = _DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] channels-first
        B, C, H, W = x.shape

        # DCNv3 residual branch (with pre-norm)
        x_ln = x.permute(0, 2, 3, 1)  # → [B, H, W, C] for LayerNorm
        x_ln = self.norm1(x_ln).permute(0, 3, 1, 2)  # → [B, C, H, W]
        x = x + self.drop_path(self.dcnv3(x_ln))

        # FFN residual branch (with pre-norm)
        x_ln = x.permute(0, 2, 3, 1)  # → [B, H, W, C]
        x_ln = self.drop_path(self.ffn(self.norm2(x_ln)))
        x = x + x_ln.permute(0, 3, 1, 2)
        return x
