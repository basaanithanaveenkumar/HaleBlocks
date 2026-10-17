"""``ACTState``: Running Adaptive Computation Time accumulators, all ``(batch, seq)`` except ``output``."""

from __future__ import annotations

from dataclasses import dataclass

import torch


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
        """Whether every position has halted (the ACT loop can stop early)."""
        return not bool(self.running.any())
