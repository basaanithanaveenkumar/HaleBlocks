"""``ConvNeXtV1Block``: Single ConvNeXt V1 residual block (no GRN, optional Layer Scale)."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.cnn.convnext_v1.stochastic_depth import StochasticDepth


class ConvNeXtV1Block(nn.Module):
    """Single ConvNeXt V1 residual block (no GRN, optional Layer Scale).

    Args:
        dim:                    Number of input/output channels.
        drop_path:              Stochastic depth probability.
        layer_scale_init_value: Initial value of per-channel layer scale γ.
                                0 disables layer scale (tiny/small/base).
    """

    def __init__(
        self,
        dim: int,
        drop_path: float = 0.0,
        layer_scale_init_value: float = 0.0,
    ) -> None:
        super().__init__()

        # Depthwise 7×7 conv: spatial mixing within each channel
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        # Inverted bottleneck: expand 4× then contract
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        # Layer Scale: per-channel learnable scalar (replaces nothing, just scales)
        self.gamma: nn.Parameter | None = None
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(torch.full((dim,), layer_scale_init_value))

        self.drop_path = StochasticDepth(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`ConvNeXtV1Block`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        residual = x

        x = self.dwconv(x)  # [B, C, H, W]
        x = x.permute(0, 2, 3, 1)  # → [B, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)  # [B, H, W, 4C]
        x = self.act(x)
        x = self.pwconv2(x)  # [B, H, W, C]

        if self.gamma is not None:
            x = x * self.gamma  # layer scale

        x = x.permute(0, 3, 1, 2)  # → [B, C, H, W]
        return residual + self.drop_path(x)
