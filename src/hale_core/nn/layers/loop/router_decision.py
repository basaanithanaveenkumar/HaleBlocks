"""``RouterDecision``: what a :class:`RecursionRouter` decided for one batch."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RouterDecision:
    """Per-token recursion depths chosen by the router.

    Attributes:
        depth: ``(batch, seq)`` long, passes each token runs (``1 .. n_loops``; 0 = inactive).
        gate: ``(batch, seq)`` scale in ``(0, 1)`` of every update the token receives (the path
            through which the router learns).
        balance_loss: scalar Switch-style load-balancing loss over the depths.
        probs: ``(batch, seq, n_loops)`` router probabilities (for inspection).
    """

    depth: torch.Tensor
    gate: torch.Tensor
    balance_loss: torch.Tensor
    probs: torch.Tensor

    def running(self, step: int) -> torch.Tensor:
        """``(batch, seq)`` bool: tokens that still recurse at 0-based pass ``step``."""
        return self.depth > step
