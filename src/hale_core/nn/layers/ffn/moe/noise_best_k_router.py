"""``NoiseBestKRouter``: Top-k expert router with optional noisy gating during training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseBestKRouter(nn.Module):
    """Top-k expert router with optional noisy gating during training."""

    def __init__(self, d_model: int, n_experts: int, top_k: int):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(d_model, n_experts)
        self.noise = nn.Linear(d_model, n_experts)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of :class:`NoiseBestKRouter`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``tuple[torch.Tensor, torch.Tensor]``.
        """
        logits = self.gate(x)
        if self.training:
            logits = logits + torch.randn_like(logits) * F.softplus(self.noise(x))
        top_logits, idx = logits.topk(self.top_k, dim=-1)
        sparse = torch.full_like(logits, float("-inf")).scatter(-1, idx, top_logits)
        return F.softmax(sparse, dim=-1), idx
