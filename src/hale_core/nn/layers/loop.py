"""Components of a looped (weight-tied, recurrent-depth) transformer.

A looped stack applies one shared block ``T`` times, ``h_t = f_W(h_{t-1})``. These modules are
what turns that raw weight tying into a trainable model:

* :class:`LoopDepthEmbedding` - the Universal Transformer's *depth* coordinate ``E^t``
  (Dehghani et al., 2019, arXiv:1807.03819), so the shared block can tell pass 1 from pass 5.
* :class:`LoopInputInjection` - re-feeds the embedded input at every pass, ``f_W(h_{t-1} + e)``
  (Yang et al., 2024, arXiv:2311.12424; Geiping et al., 2025, arXiv:2502.05171). Without it the
  input is only visible at ``t = 0`` and long loops forget it.
* :class:`ACTHalting` - Adaptive Computation Time (Graves, 2016, arXiv:1603.08983): a learned
  per-token halting probability; each token's output is the halting-weighted mixture of its
  states and a ponder cost ``N(i) + R(i)`` penalises extra passes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn

DepthEmbeddingKind = Literal["none", "sinusoidal", "learned"]
InjectionKind = Literal["none", "add", "concat"]


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


class LoopInputInjection(nn.Module):
    """Combine the recurrent state with the loop input ``e`` before each pass.

    ``add``: ``h + e``. ``concat``: ``W [h; e]`` (Huginn's adapter), initialised so that it starts
    as ``h + e``. ``none``: ``h`` (the input enters only through ``h_0 = e``).
    """

    def __init__(self, kind: InjectionKind, d_model: int) -> None:
        super().__init__()
        if kind not in ("none", "add", "concat"):
            raise ValueError(f"unknown input injection {kind!r}")
        self.kind = kind
        self.adapter = nn.Linear(2 * d_model, d_model, bias=False) if kind == "concat" else None
        if self.adapter is not None:
            with torch.no_grad():
                eye = torch.eye(d_model)
                self.adapter.weight.copy_(torch.cat([eye, eye], dim=1))

    @property
    def starts_from_zero(self) -> bool:
        """Injected loops start from ``h_0 = 0``; the input arrives through the injection."""
        return self.kind != "none"

    def forward(self, state: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
        if self.kind == "none":
            return state
        if self.kind == "add":
            return state + inputs
        return self.adapter(torch.cat([state, inputs], dim=-1))


@dataclass
class ACTState:
    """Running Adaptive Computation Time accumulators, all ``(batch, seq)`` except ``output``.

    Attributes:
        halting: accumulated halting probability of each token.
        remainders: ``R(i)``, the probability mass left when the token halted.
        n_updates: ``N(i)``, the number of passes that contributed to the token's output.
        running: tokens still computing.
        output: ``(batch, seq, d_model)`` halting-weighted mixture of the states so far.
    """

    halting: torch.Tensor
    remainders: torch.Tensor
    n_updates: torch.Tensor
    running: torch.Tensor
    output: torch.Tensor

    @property
    def all_halted(self) -> bool:
        return not bool(self.running.any())


class ACTHalting(nn.Module):
    """Per-token halting unit ``p_t^i = sigmoid(w . h_t^i + b)`` and its ACT bookkeeping.

    Follows the Universal Transformer recipe: a token halts at the first step where its
    cumulative halting probability would exceed ``threshold`` (``1 - eps``); that step gets the
    remainder ``R = 1 - sum p``. The output of every token is ``sum_t lambda_t h_t`` with
    ``lambda_t = p_t`` before halting and ``R`` at the halting step. Halted tokens keep being
    transformed (other tokens attend to them) but no longer change their own output.

    Args:
        d_model: hidden width.
        threshold: ``1 - eps``.
        init_bias: initial halting logit; negative values start the model "pondering" (Graves
            uses a positive bias for early halting; the UT uses 1.0 - a choice of budget).
    """

    def __init__(self, d_model: int, threshold: float = 0.99, init_bias: float = 1.0) -> None:
        super().__init__()
        if not 0.0 < threshold <= 1.0:
            raise ValueError(f"ACT threshold must be in (0, 1], got {threshold}")
        self.threshold = threshold
        self.proj = nn.Linear(d_model, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, init_bias)

    def init_state(self, hidden: torch.Tensor, active: torch.Tensor) -> ACTState:
        """Fresh accumulators; tokens with ``active == False`` (padding, scratch) never run."""
        zeros = hidden.new_zeros(hidden.shape[:2], dtype=torch.float32)
        return ACTState(
            zeros, zeros.clone(), zeros.clone(), active.clone(), torch.zeros_like(hidden)
        )

    def step(self, state: ACTState, hidden: torch.Tensor, last: bool) -> ACTState:
        """Fold the pass's output ``hidden`` into ``state``; ``last`` forces every token to halt."""
        p = torch.sigmoid(self.proj(hidden).squeeze(-1).float())
        running = state.running.float()
        would_halt = (state.halting + p * running) > self.threshold
        if last:
            would_halt = torch.ones_like(would_halt)
        new_halted = would_halt.float() * running
        still = (~would_halt).float() * running
        halting = state.halting + p * still
        remainder = new_halted * (1.0 - halting)  # nonzero only for tokens halting at this pass
        n_updates = state.n_updates + still + new_halted
        weight = (p * still + remainder).unsqueeze(-1)
        output = state.output + weight.to(hidden.dtype) * hidden
        remainders = state.remainders + remainder
        halting = halting + remainder
        return ACTState(halting, remainders, n_updates, still.bool(), output)

    @staticmethod
    def ponder_cost(state: ACTState, active: torch.Tensor) -> torch.Tensor:
        """Mean ``N(i) + R(i)`` over the ``active`` tokens."""
        mask = active.float()
        return ((state.n_updates + state.remainders) * mask).sum() / mask.sum().clamp_min(1.0)
