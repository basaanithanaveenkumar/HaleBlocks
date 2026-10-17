"""``SwiGLUFFN``: SwiGLU gated feed-forward network."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.vit.vit import _trunc_normal_


class SwiGLUFFN(nn.Module):
    """SwiGLU gated feed-forward network.

    Computes:  SiLU(W₁·x) ⊙ W₂·x → W₃

    The intermediate dim is 8/3 × dim (not 4×) so that the total parameter
    count matches a standard 4× GELU MLP (two projections instead of one).

    Reference: "GLU Variants Improve Transformer" (Shazeer, 2020)
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        # Scale hidden dim by 2/3 to keep total params equal to GELU MLP at same ratio
        hidden = int(dim * mlp_ratio * 2 / 3)
        # Make divisible by 256 for efficient GEMM
        hidden = (hidden + 255) // 256 * 256

        self.w1 = nn.Linear(dim, hidden, bias=False)  # gate branch (SiLU)
        self.w2 = nn.Linear(dim, hidden, bias=False)  # value branch
        self.w3 = nn.Linear(hidden, dim, bias=False)  # output projection
        self.drop = nn.Dropout(drop)

        _trunc_normal_(self.w1.weight)
        _trunc_normal_(self.w2.weight)
        _trunc_normal_(self.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`SwiGLUFFN`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        # SiLU gate applied element-wise to the value branch
        gate = F.silu(self.w1(x))
        value = self.w2(x)
        return self.drop(self.w3(gate * value))
