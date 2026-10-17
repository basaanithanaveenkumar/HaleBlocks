"""``LoopedStack``: Prelude, a weight-tied core block looped ``T`` times, coda (see the module docstring)."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.layers import TransformerLayer, build_attention, build_ffn
from hale_core.nn.layers.loop import (
    ACTHalting,
    LoopDepthEmbedding,
    LoopInputInjection,
)
from hale_core.nn.stacks.looped.loop_spec import LoopSpec
from hale_core.nn.stacks.looped.loop_stats import LoopStats


class LoopedStack(nn.Module):
    """Prelude, a weight-tied core block looped ``T`` times, coda (see the module docstring).

    ``self.layers`` holds ``prelude + core + coda`` in that order (the names LoRA / FSDP see);
    :attr:`prelude`, :attr:`core` and :attr:`coda` are views on it.

    Args:
        d_model / n_heads / d_ff / dropout / attn_impl / n_kv_heads / ffn_type / moe_*: as in
            :class:`~hale_core.nn.stacks.GPTStack`, for every layer.
        n_layers: layers in the shared core block.
        use_time_cond: AdaLN-style diffusion / flow time conditioning in every layer.
        loop: :class:`LoopSpec` (or anything :meth:`LoopSpec.coerce` accepts).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        dropout: float = 0.0,
        attn_impl: str = "mha",
        n_kv_heads: int | None = None,
        ffn_type: str = "mlp",
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        moe_num_shared: int = 1,
        use_time_cond: bool = False,
        loop: LoopSpec | dict | None = None,
    ):
        super().__init__()
        self.spec = spec = LoopSpec.coerce(loop)
        if n_layers < 1:
            raise ValueError(f"the looped core block needs n_layers >= 1, got {n_layers}")

        def layer(post_norm: bool) -> TransformerLayer:
            return TransformerLayer(
                d_model,
                attention=build_attention(
                    attn_impl,
                    d_model=d_model,
                    n_heads=n_heads,
                    dropout=dropout,
                    n_kv_heads=n_kv_heads,
                ),
                ffn=build_ffn(
                    ffn_type,
                    d_model=d_model,
                    d_ff=d_ff,
                    dropout=dropout,
                    moe_num_experts=moe_num_experts,
                    moe_top_k=moe_top_k,
                    moe_num_shared=moe_num_shared,
                ),
                use_time_cond=use_time_cond,
                norm="layer",
                post_norm=post_norm,
            )

        self.layers = nn.ModuleList(
            [layer(False) for _ in range(spec.prelude_layers)]
            + [layer(spec.sandwich_norm) for _ in range(n_layers)]
            + [layer(False) for _ in range(spec.coda_layers)]
        )
        self.n_core_layers = n_layers
        self.depth_embedding = LoopDepthEmbedding(spec.depth_embedding, d_model, spec.n_loops)
        self.injection = LoopInputInjection(spec.input_injection, d_model)
        self.halting = (
            ACTHalting(d_model, spec.act_threshold, spec.act_init_bias)
            if spec.halting == "act"
            else None
        )
        self.scratch = (
            nn.Parameter(torch.randn(spec.scratch_tokens, d_model) * 0.02)
            if spec.scratch_tokens
            else None
        )
        self.ln_out = nn.LayerNorm(d_model)
        #: Loop count used outside training (``None`` = ``spec.n_loops``); see :func:`loop_budget`.
        self.eval_loops: int | None = None
        self.last_stats: LoopStats | None = None

    # ------------------------------------------------------------------ structure
    @property
    def prelude(self) -> nn.ModuleList:
        """Layers run once before the loop."""
        return self.layers[: self.spec.prelude_layers]

    @property
    def core(self) -> nn.ModuleList:
        """Weight-tied layers iterated by the loop."""
        p = self.spec.prelude_layers
        return self.layers[p : p + self.n_core_layers]

    @property
    def coda(self) -> nn.ModuleList:
        """Layers run once after the loop."""
        return self.layers[self.spec.prelude_layers + self.n_core_layers :]

    @property
    def effective_depth(self) -> int:
        """Layers applied per forward at the default loop count."""
        return (
            self.spec.prelude_layers
            + self.spec.n_loops * self.n_core_layers
            + self.spec.coda_layers
        )

    @property
    def aux_loss(self) -> torch.Tensor | None:
        """``ponder_weight * ponder_cost`` of the last forward (ACT only), for the task loss."""
        stats = self.last_stats
        if stats is None or stats.ponder_cost is None:
            return None
        return self.spec.ponder_weight * stats.ponder_cost

    def num_loops(self) -> int:
        """Passes for the next forward: sampled in training when ``min_loops`` is set."""
        spec = self.spec
        if self.training:
            if spec.min_loops is None or spec.min_loops == spec.n_loops:
                return spec.n_loops
            return int(torch.randint(spec.min_loops, spec.n_loops + 1, ()).item())
        return self.eval_loops or spec.n_loops

    # ------------------------------------------------------------------ forward
    def forward(
        self,
        h: torch.Tensor,
        t_emb: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass of :class:`LoopedStack`.

        Args:
            h: input (``torch.Tensor``).
            t_emb: input (``torch.Tensor | None``).
            attn_mask: input (``torch.Tensor | None``).
            key_padding_mask: input (``torch.Tensor | None``).
            positions: input (``torch.Tensor | None``).

        Returns:
            ``torch.Tensor``.
        """
        kw = {
            "t_emb": t_emb,
            "attn_mask": attn_mask,
            "key_padding_mask": key_padding_mask,
            "positions": positions,
        }
        for layer in self.prelude:
            h = layer(h, **kw)

        n_scratch = self.spec.scratch_tokens
        if n_scratch:
            h, kw = self._prepend_scratch(h, kw)
        active = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        if kw["key_padding_mask"] is not None:
            active &= ~kw["key_padding_mask"]
        active[:, :n_scratch] = False

        state = self._loop(h, active, kw)
        for layer in self.coda:
            state = layer(state, **kw)
        return self.ln_out(state[:, n_scratch:])

    def _loop(self, inputs: torch.Tensor, active: torch.Tensor, kw: dict) -> torch.Tensor:
        spec = self.spec
        n_loops = self.num_loops()
        grad_from = n_loops - spec.bptt_loops if self.training and spec.bptt_loops else 0
        state = torch.zeros_like(inputs) if self.injection.starts_from_zero else inputs
        act = self.halting.init_state(state, active) if self.halting is not None else None
        check_exit = spec.exit_tolerance is not None and not self.training and act is None
        residual = None
        loops = 0
        for step in range(n_loops):
            previous = state
            with torch.set_grad_enabled(torch.is_grad_enabled() and step >= grad_from):
                x = self.injection(state, inputs)
                depth = self.depth_embedding(step, x)
                if depth is not None:
                    x = x + depth
                for layer in self.core:
                    x = layer(x, **kw)
            state = x
            loops = step + 1
            if act is not None:
                act = self.halting.step(act, state, last=loops == n_loops)
                if act.all_halted:
                    break
            elif check_exit:
                residual = self._relative_change(state, previous, active)
                if residual < spec.exit_tolerance:
                    break

        if act is None:
            self.last_stats = LoopStats(loops=loops, residual=residual)
            return state
        real = active.float()
        self.last_stats = LoopStats(
            loops=loops,
            ponder_cost=ACTHalting.ponder_cost(act, active),
            mean_updates=float((act.n_updates * real).sum() / real.sum().clamp_min(1.0)),
            token_updates=act.n_updates.detach()[:, self.spec.scratch_tokens :],
        )
        return torch.where(active.unsqueeze(-1), act.output, state)

    # ------------------------------------------------------------------ helpers
    def _prepend_scratch(self, h: torch.Tensor, kw: dict) -> tuple[torch.Tensor, dict]:
        """Prepend the scratch slots and grow every mask / position input to match."""
        b, n, s = h.size(0), h.size(1), self.spec.scratch_tokens
        h = torch.cat([self.scratch.to(h.dtype).unsqueeze(0).expand(b, -1, -1), h], dim=1)
        out = dict(kw)
        mask = kw["attn_mask"]
        if mask is not None:
            if mask.dim() != 2:
                raise ValueError("scratch tokens need a 2-D (seq, seq) attention mask")
            blocked = True if mask.dtype == torch.bool else float("-inf")
            grown = mask.new_zeros(s + n, s + n)
            grown[s:, s:] = mask
            grown[:s, s:] = blocked  # scratch must not read tokens a causal row may not see
            out["attn_mask"] = grown
        padding = kw["key_padding_mask"]
        if padding is not None:
            out["key_padding_mask"] = torch.cat([padding.new_zeros(b, s), padding], dim=1)
        positions = kw["positions"]
        if positions is not None:
            out["positions"] = torch.cat([positions.new_zeros(s), positions])
        return h, out

    @staticmethod
    def _relative_change(
        state: torch.Tensor, previous: torch.Tensor, active: torch.Tensor
    ) -> float:
        state, previous = state.detach().float(), previous.detach().float()
        change = (state - previous).norm(dim=-1) / previous.norm(dim=-1).clamp_min(1e-6)
        change = change.masked_fill(~active, 0.0)
        return float(change.max())
