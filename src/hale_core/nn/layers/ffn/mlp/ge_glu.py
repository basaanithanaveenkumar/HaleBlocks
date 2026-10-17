"""``GeGLU``: Gated GELU FFN used by the LGT (local-global transformer) decoder."""

from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class GeGLU(nn.Module):
    """Gated GELU FFN used by the LGT (local-global transformer) decoder."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.w_in = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w_out = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass of :class:`GeGLU`.

        Args:
            x: input.
        """
        gate, up = self.w_in(x).chunk(2, dim=-1)
        return self.w_out(self.drop(F.gelu(gate) * up))
