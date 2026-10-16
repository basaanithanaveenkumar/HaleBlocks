"""``LoopStats``: What the last forward of a :class:`LoopedStack` did."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LoopStats:
    """What the last forward of a :class:`LoopedStack` did.

    Attributes:
        loops: passes actually run (``< T`` after an ACT or fixed-point early exit).
        ponder_cost: mean ``N(i) + R(i)`` over real tokens (ACT only; carries gradient).
        mean_updates: mean ``N(i)``, the average passes per token (ACT only).
        residual: largest relative change of the last pass (when ``exit_tolerance`` is set).
        token_updates: ``(batch, seq)`` passes each token used, ``N(i)`` (ACT only; detached,
            scratch slots removed, 0 at padding).
    """

    loops: int
    ponder_cost: torch.Tensor | None = None
    mean_updates: float | None = None
    residual: float | None = None
    token_updates: torch.Tensor | None = None
