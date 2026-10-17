"""``DiTBlock``: DiT block: pre-norm attention + FFN modulated by AdaLN-Zero time conditioning."""

from __future__ import annotations

import torch
import torch.nn as nn


class DiTBlock(nn.Module):
    """DiT block: pre-norm attention + FFN modulated by AdaLN-Zero time conditioning."""
    def __init__(
        self,
        d_model: int,
        attention: nn.Module,
        ffn: nn.Module,
        *,
        norm: str = "layer",
        use_time_cond: bool = True,
        post_norm: bool = False,
    ):
        super().__init__()
        self.attn = attention
        self.ff = ffn
        # Build norm layers via registry so "rms" is supported
        from hale_core.registry.plugins import NORMS_REGISTRY

        norm_cls = NORMS_REGISTRY.get(norm) if norm in NORMS_REGISTRY else nn.LayerNorm
        self.ln1 = norm_cls(d_model)
        self.ln2 = norm_cls(d_model)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 6 * d_model))
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def _mod(self, t_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        chunks = self.adaLN_modulation(t_emb).chunk(6, dim=-1)
        if chunks[0].dim() == 2:
            return tuple(c.unsqueeze(1) for c in chunks)
        return chunks

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of :class:`DiTBlock`.

        Args:
            x: input (``torch.Tensor``).
            t_emb: input (``torch.Tensor | None``).
            attn_mask: input (``torch.Tensor | None``).
            key_padding_mask: input (``torch.Tensor | None``).
            positions: input (``torch.Tensor | None``).

        Returns:
            ``torch.Tensor``.
        """
        if t_emb is None:
            t_emb = torch.zeros(x.size(0), x.size(-1), device=x.device, dtype=x.dtype)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self._mod(t_emb)
        h = self.ln1(x) * (1 + scale_msa) + shift_msa
        x = x + gate_msa * self.attn(
            h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, positions=positions
        )
        h = self.ln2(x) * (1 + scale_mlp) + shift_mlp
        return x + gate_mlp * self.ff(h)
