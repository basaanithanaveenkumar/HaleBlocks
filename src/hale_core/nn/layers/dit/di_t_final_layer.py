"""``DiTFinalLayer``: DiTFinalLayer (DiT block (Peebles & Xie): pre-norm attention + FFN with AdaLN-Zero)."""

from __future__ import annotations

import torch
import torch.nn as nn


class DiTFinalLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 2 * d_model))
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor | None) -> torch.Tensor:
        if t_emb is None:
            t_emb = torch.zeros(x.size(0), x.size(-1), device=x.device, dtype=x.dtype)
        shift, scale = self.adaLN_modulation(t_emb).chunk(2, dim=-1)
        if shift.dim() == 2:
            shift, scale = shift.unsqueeze(1), scale.unsqueeze(1)
        return self.head(self.ln(x) * (1 + scale) + shift)
