"""``ACTHalting``: Per-token halting unit ``p_t^i = sigmoid(w . h_t^i + b)`` and its ACT bookkeeping."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.layers.loop.act_state import ACTState


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
