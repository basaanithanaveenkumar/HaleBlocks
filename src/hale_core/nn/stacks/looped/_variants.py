"""Variants and registrations of :mod:`hale_core.nn.stacks.looped` (Looped (recurrent-depth) transformer stack: one shared block applied ``T`` times)."""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import torch
import torch.nn as nn

from hale_core.nn.stacks.looped.looped_stack import LoopedStack


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
