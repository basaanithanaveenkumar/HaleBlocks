"""``DeepseekMoE``: DeepseekMoE (DeepSeek-style MoE: noisy top-k router + shared experts (HaloBlocks / HALO-WAM))."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.layers.ffn.moe.noise_best_k_router import NoiseBestKRouter
from hale_core.nn.layers.ffn.moe.swi_glu_expert import SwiGLUExpert


class DeepseekMoE(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        n_experts: int = 4,
        top_k: int = 2,
        n_shared: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        if top_k > n_experts:
            raise ValueError(f"top_k={top_k} > n_experts={n_experts}")
        self.router = NoiseBestKRouter(d_model, n_experts, top_k)
        self.shared = nn.ModuleList([SwiGLUExpert(d_model, d_ff, dropout) for _ in range(n_shared)])
        self.routed = nn.ModuleList(
            [SwiGLUExpert(d_model, d_ff, dropout) for _ in range(n_experts)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        flat = x.reshape(-1, d)
        shared = sum((expert(flat) for expert in self.shared), start=torch.zeros_like(flat))
        gates, idx = self.router(x)
        out = torch.zeros_like(x)
        flat_gates = gates.reshape(-1, gates.size(-1))
        for i, expert in enumerate(self.routed):
            token_mask = (idx == i).any(dim=-1)
            flat_mask = token_mask.reshape(-1)
            if not flat_mask.any():
                continue
            weighted = expert(flat[flat_mask]) * flat_gates[flat_mask, i].unsqueeze(1)
            out[token_mask] = out[token_mask] + weighted
        return out + shared.view(b, s, d)
