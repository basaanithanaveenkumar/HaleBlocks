"""``LoopLoRALinear``: a shared linear layer plus one low-rank delta per loop pass."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoopLoRALinear(nn.Module):
    """``y = W x + b + B_t A_t x`` where ``t`` is the current loop pass.

    The *relaxed recursive transformer* (Bae et al., 2024, arXiv:2410.20672) keeps one shared
    weight ``W`` for every pass of the loop but gives each pass ``t`` its own rank-``r`` delta
    ``B_t A_t``. Strict weight tying becomes a soft constraint: passes can specialise a little
    (pass 1 parses, the last pass extracts) for ``O(T * r * (d_in + d_out))`` extra parameters.

    ``B_t`` starts at zero, so a freshly built layer is exactly the tied layer. The stack sets
    :attr:`pass_index` before every pass; passes beyond the trained ``n_loops`` (test-time loop
    budgets) reuse the last delta, and ``pass_index = None`` disables the delta.

    Args:
        base: the shared ``nn.Linear`` (kept as ``self.base``, so its weights stay tied).
        n_loops: number of per-pass deltas.
        rank: LoRA rank ``r``.
    """

    def __init__(self, base: nn.Linear, n_loops: int, rank: int) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError(f"LoRA rank must be >= 1, got {rank}")
        self.base = base
        self.rank = rank
        self.down = nn.Parameter(
            torch.randn(n_loops, rank, base.in_features) / math.sqrt(base.in_features)
        )
        self.up = nn.Parameter(torch.zeros(n_loops, base.out_features, rank))
        #: Index of the pass being run (set by :class:`~hale_core.nn.stacks.looped.LoopedStack`).
        self.pass_index: int | None = None

    @property
    def in_features(self) -> int:
        return self.base.in_features

    @property
    def out_features(self) -> int:
        return self.base.out_features

    @property
    def weight(self) -> torch.Tensor:
        """The shared weight (so code that reads ``.weight`` keeps working)."""
        return self.base.weight

    def delta_weight(self, pass_index: int) -> torch.Tensor:
        """``B_t A_t`` - the ``(out, in)`` weight offset of pass ``t``."""
        t = min(pass_index, self.down.shape[0] - 1)
        return self.up[t] @ self.down[t]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Shared projection plus the current pass's low-rank delta."""
        out = self.base(x)
        if self.pass_index is None:
            return out
        t = min(self.pass_index, self.down.shape[0] - 1)
        return out + (x @ self.down[t].transpose(0, 1).to(x.dtype)) @ self.up[t].transpose(0, 1).to(x.dtype)

    @staticmethod
    def wrap(module: nn.Module, n_loops: int, rank: int) -> int:
        """Replace every ``nn.Linear`` inside ``module`` by a :class:`LoopLoRALinear`.

        Linear layers that belong to ``nn.MultiheadAttention`` are left alone: PyTorch's fused
        attention reads their weights directly and never calls them, so a wrapper would be
        bypassed (use a ``gqa`` / ``mqa`` attention to give attention projections per-pass deltas).

        Returns:
            the number of wrapped layers.
        """
        count = 0
        for name, child in list(module.named_children()):
            if isinstance(child, nn.MultiheadAttention | LoopLoRALinear):
                continue
            if isinstance(child, nn.Linear):
                setattr(module, name, LoopLoRALinear(child, n_loops, rank))
                count += 1
            else:
                count += LoopLoRALinear.wrap(child, n_loops, rank)
        return count

    @staticmethod
    def set_pass(module: nn.Module, pass_index: int | None) -> None:
        """Point every :class:`LoopLoRALinear` inside ``module`` at pass ``pass_index``."""
        for child in module.modules():
            if isinstance(child, LoopLoRALinear):
                child.pass_index = pass_index
