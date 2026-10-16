"""Built-in LR scheduler factories — registered in SCHEDULERS_REGISTRY.

Each factory has the signature::

    factory(optimizer, cfg: SchedulerConfig, total_steps: int) -> LRScheduler

Import this module as a side effect to register all built-ins.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch.optim as optim

    from hale_core.config.sections.schedule import SchedulerConfig


def _constant(
    optimizer: optim.Optimizer,
    cfg: SchedulerConfig,
    total_steps: int,
) -> Any:
    from torch.optim.lr_scheduler import ConstantLR

    return ConstantLR(optimizer, factor=1.0, total_iters=total_steps)


def _cosine(
    optimizer: optim.Optimizer,
    cfg: SchedulerConfig,
    total_steps: int,
) -> Any:
    from torch.optim.lr_scheduler import CosineAnnealingLR

    base_lr = optimizer.param_groups[0]["lr"]
    eta_min = base_lr * cfg.min_lr_ratio
    t_max = cfg.t_max if cfg.t_max is not None else total_steps
    return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)


def _linear(
    optimizer: optim.Optimizer,
    cfg: SchedulerConfig,
    total_steps: int,
) -> Any:
    from torch.optim.lr_scheduler import LinearLR

    return LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=cfg.min_lr_ratio,
        total_iters=total_steps,
    )


def _warmup_cosine(
    optimizer: optim.Optimizer,
    cfg: SchedulerConfig,
    total_steps: int,
) -> Any:
    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    warmup_steps = min(cfg.warmup_steps, total_steps)
    decay_steps = max(total_steps - warmup_steps, 1)
    base_lr = optimizer.param_groups[0]["lr"]
    eta_min = base_lr * cfg.min_lr_ratio

    warmup = LinearLR(
        optimizer,
        start_factor=cfg.warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    cosine = CosineAnnealingLR(optimizer, T_max=decay_steps, eta_min=eta_min)
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


def _one_cycle(
    optimizer: optim.Optimizer,
    cfg: SchedulerConfig,
    total_steps: int,
) -> Any:
    from torch.optim.lr_scheduler import OneCycleLR

    max_lr = [pg["lr"] for pg in optimizer.param_groups]
    return OneCycleLR(
        optimizer,
        max_lr=max_lr,
        total_steps=total_steps,
        pct_start=cfg.pct_start,
        div_factor=cfg.div_factor,
        final_div_factor=cfg.final_div_factor,
    )


# ---------------------------------------------------------------------------
# Registration (side-effect on import)
# ---------------------------------------------------------------------------


def _register_schedulers() -> None:
    from hale_core.registry.plugins import SCHEDULERS_REGISTRY

    entries = {
        "constant": _constant,
        "cosine": _cosine,
        "linear": _linear,
        "warmup_cosine": _warmup_cosine,
        "one_cycle": _one_cycle,
    }
    for name, fn in entries.items():
        if name not in SCHEDULERS_REGISTRY:
            SCHEDULERS_REGISTRY.add(name, fn)


_register_schedulers()
