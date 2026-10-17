"""``LayerScale``: Per-channel learned scale applied to a residual branch."""

from __future__ import annotations

import torch
import torch.nn as nn

# Reuse PatchEmbed and MLP from sibling vit module (avoid duplication)


class LayerScale(nn.Module):
    """Per-channel learned scale applied to a residual branch.

    Initialised at `init_value` (≈1e-5 for large models) so residuals start
    near zero.  The scale is learned and grows during training, allowing the
    network to gradually increase the residual contribution.

    Reference: "Going Deeper with Image Transformers" (Touvron et al., 2021)
               https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim: int, init_value: float = 1e-5) -> None:
        super().__init__()
        # Scalar per output channel; starts small to dampen residuals initially
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`LayerScale`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        return x * self.gamma
