"""Looped (recurrent-depth) transformer stack: one shared block applied ``T`` times.

Layout (``prelude -> T x core -> coda``, Geiping et al., 2025, arXiv:2502.05171)::

    e   = prelude(x)                         distinct layers, run once (optional)
    h_0 = 0 (with input injection) or e
    h_t = core(inject(h_{t-1}, e) + E^t)     the same weights for t = 1 .. T
    out = coda(h_T  or  ACT mixture of h_t)  distinct layers, run once (optional)

``core`` is ``n_layers`` pre-norm transformer layers; the parameter count is that of
``prelude + core + coda`` while the compute depth is ``prelude + T * core + coda``. Supported
variants (all through :class:`LoopSpec`):

* fixed-``T`` looping with a depth embedding (Universal Transformer, arXiv:1807.03819);
* ACT-gated looping - tokens halt at different passes, ponder cost exposed as ``aux_loss``;
* randomised ``T`` during training and truncated backprop through the last ``k`` passes - the
  recurrent-depth recipe that makes the loop count a test-time-compute knob;
* fixed-point early exit at inference (``h_t ~= h_{t-1}``, the DEQ view, arXiv:1909.01377);
* scratchpad tokens - learned working-memory slots carried across passes and dropped from the
  output.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

import torch
import torch.nn as nn

from hale_core.nn.layers import TransformerLayer, build_attention, build_ffn
from hale_core.nn.layers.loop import (
    ACTHalting,
    DepthEmbeddingKind,
    InjectionKind,
    LoopDepthEmbedding,
    LoopInputInjection,
)


@dataclass(frozen=True)
class LoopSpec:
    """How a :class:`LoopedStack` iterates its shared block.

    Attributes:
        n_loops: passes through the shared block (training maximum and inference default).
        prelude_layers: distinct layers before the loop (map tokens into the latent space).
        coda_layers: distinct layers after the loop (map the latent state back to tokens).
        depth_embedding: loop-step coordinate added to each pass's input
            (:class:`LoopDepthEmbedding`).
        input_injection: how the loop input re-enters every pass (:class:`LoopInputInjection`).
        sandwich_norm: add post-norms inside each core layer (keeps ``h_t`` bounded over long
            loops, i.e. keeps ``I + J_f`` well conditioned).
        halting: ``none`` = run ``T`` passes; ``act`` = Adaptive Computation Time per token.
        act_threshold: ``1 - eps`` of ACT.
        act_init_bias: initial halting logit (positive halts early).
        ponder_weight: ``tau``; ``aux_loss = tau * mean(N + R)`` when ``halting == "act"``.
        min_loops: training only - sample ``T ~ U{min_loops .. n_loops}`` every forward so the
            model works for any loop count (``None`` = always ``n_loops``).
        bptt_loops: training only - backpropagate through the last ``k`` passes; earlier passes
            run without autograd (``None`` = full backprop through time). Memory becomes
            ``O(k)`` instead of ``O(T)``.
        exit_tolerance: inference only - stop once the largest per-token relative change
            ``|h_t - h_{t-1}| / |h_{t-1}|`` drops below this (``None`` = always run ``T``).
        scratch_tokens: learned memory tokens prepended to the sequence for the stack only.
            Every token reads them; under a causal / block mask they read only each other (so no
            future token leaks), with bidirectional attention they read the whole sequence.
    """

    n_loops: int = 4
    prelude_layers: int = 0
    coda_layers: int = 0
    depth_embedding: DepthEmbeddingKind = "sinusoidal"
    input_injection: InjectionKind = "add"
    sandwich_norm: bool = False
    halting: Literal["none", "act"] = "none"
    act_threshold: float = 0.99
    act_init_bias: float = 1.0
    ponder_weight: float = 0.01
    min_loops: int | None = None
    bptt_loops: int | None = None
    exit_tolerance: float | None = None
    scratch_tokens: int = 0

    def __post_init__(self) -> None:
        if self.n_loops < 1:
            raise ValueError(f"n_loops must be >= 1, got {self.n_loops}")
        if self.prelude_layers < 0 or self.coda_layers < 0 or self.scratch_tokens < 0:
            raise ValueError("prelude_layers, coda_layers and scratch_tokens must be >= 0")
        if self.min_loops is not None and not 1 <= self.min_loops <= self.n_loops:
            raise ValueError(
                f"min_loops must be in [1, n_loops={self.n_loops}], got {self.min_loops}"
            )
        if self.bptt_loops is not None and self.bptt_loops < 1:
            raise ValueError(f"bptt_loops must be >= 1, got {self.bptt_loops}")
        if self.halting not in ("none", "act"):
            raise ValueError(f"unknown halting {self.halting!r}")
        if self.exit_tolerance is not None and self.exit_tolerance <= 0:
            raise ValueError(f"exit_tolerance must be > 0, got {self.exit_tolerance}")

    @classmethod
    def coerce(cls, value: Any) -> LoopSpec:
        """A spec from ``None``, a mapping, a pydantic section or a spec."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        elif not isinstance(value, dict):
            value = vars(value)
        known = {f.name for f in fields(cls)}
        unknown = set(value) - known
        if unknown:
            raise ValueError(f"unknown loop options {sorted(unknown)}; expected {sorted(known)}")
        return cls(**value)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LoopStats:
    """What the last forward of a :class:`LoopedStack` did.

    Attributes:
        loops: passes actually run (``< T`` after an ACT or fixed-point early exit).
        ponder_cost: mean ``N(i) + R(i)`` over real tokens (ACT only; carries gradient).
        mean_updates: mean ``N(i)``, the average passes per token (ACT only).
        residual: largest relative change of the last pass (when ``exit_tolerance`` is set).
    """

    loops: int
    ponder_cost: torch.Tensor | None = None
    mean_updates: float | None = None
    residual: float | None = None


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
        return self.layers[: self.spec.prelude_layers]

    @property
    def core(self) -> nn.ModuleList:
        p = self.spec.prelude_layers
        return self.layers[p : p + self.n_core_layers]

    @property
    def coda(self) -> nn.ModuleList:
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


def looped_stacks(module: nn.Module) -> list[LoopedStack]:
    """Every :class:`LoopedStack` inside ``module``."""
    return [m for m in module.modules() if isinstance(m, LoopedStack)]


def set_loop_budget(module: nn.Module, n_loops: int | None) -> int:
    """Run every looped stack in ``module`` for ``n_loops`` passes outside training.

    More loops than training is test-time compute scaling; ``None`` restores ``spec.n_loops``.
    Returns the number of stacks changed (0 = the model has no looped stack).
    """
    if n_loops is not None and n_loops < 1:
        raise ValueError(f"n_loops must be >= 1, got {n_loops}")
    stacks = looped_stacks(module)
    for stack in stacks:
        stack.eval_loops = n_loops
    return len(stacks)


@contextlib.contextmanager
def loop_budget(module: nn.Module, n_loops: int | None) -> Iterator[None]:
    """Temporarily set the inference loop count (see :func:`set_loop_budget`)."""
    previous = [(s, s.eval_loops) for s in looped_stacks(module)]
    set_loop_budget(module, n_loops)
    try:
        yield
    finally:
        for stack, value in previous:
            stack.eval_loops = value


def loop_aux_loss(module: nn.Module) -> torch.Tensor | None:
    """Sum of the ACT ponder losses of the last forward of every looped stack in ``module``."""
    losses = [s.aux_loss for s in looped_stacks(module) if s.aux_loss is not None]
    return torch.stack(losses).sum() if losses else None


def loop_stats(module: nn.Module) -> dict[str, float]:
    """Scalar loop statistics of the last forward (``loop/loops``, ``loop/ponder`` ...)."""
    out: dict[str, float] = {}
    for i, stack in enumerate(looped_stacks(module)):
        stats = stack.last_stats
        if stats is None:
            continue
        prefix = "loop" if i == 0 else f"loop{i}"
        out[f"{prefix}/loops"] = float(stats.loops)
        if stats.ponder_cost is not None:
            out[f"{prefix}/ponder_cost"] = float(stats.ponder_cost.detach())
        if stats.mean_updates is not None:
            out[f"{prefix}/mean_updates"] = stats.mean_updates
        if stats.residual is not None:
            out[f"{prefix}/residual"] = stats.residual
    return out
