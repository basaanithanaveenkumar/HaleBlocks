"""``LoopDepthEmbedding``: Embedding of the loop step ``t`` (0-based), added to every token's input of that pass."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from hale_core.nn.layers.loop._helpers import DepthEmbeddingKind


class LoopDepthEmbedding(nn.Module):
    """Embedding of the loop step ``t`` (0-based), added to every token's input of that pass.

    Args:
        kind: ``sinusoidal`` (fixed, defined for any ``t`` - lets inference run more loops than
            training), ``learned`` (one vector per step up to ``max_steps``) or ``none``.
        d_model: hidden width.
        max_steps: number of rows of the ``learned`` table.
    """

    def __init__(self, kind: DepthEmbeddingKind, d_model: int, max_steps: int) -> None:
        super().__init__()
        if kind not in ("none", "sinusoidal", "learned"):
            raise ValueError(f"unknown depth embedding {kind!r}")
        self.kind = kind
        self.d_model = d_model
        self.max_steps = max_steps
        self.table = nn.Embedding(max_steps, d_model) if kind == "learned" else None
        if self.table is not None:
            nn.init.normal_(self.table.weight, std=0.02)

    def forward(self, step: int, like: torch.Tensor) -> torch.Tensor | None:
        """``(d_model,)`` embedding of ``step`` (dtype / device of ``like``); ``None`` = off."""
        if self.kind == "none":
            return None
        if self.table is not None:
            if step >= self.max_steps:
                raise ValueError(
                    f"loop step {step} has no learned depth embedding "
                    f"(max_steps={self.max_steps}); "
                    "use depth_embedding='sinusoidal' to run more loops at inference"
                )
            return self.table.weight[step].to(like.dtype)
        half = self.d_model // 2
        freqs = torch.exp(
            -math.log(10_000.0) * torch.arange(half, device=like.device).float() / half
        )
        args = float(step + 1) * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)])
        if self.d_model % 2:
            emb = torch.cat([emb, emb.new_zeros(1)])
        return emb.to(like.dtype)
