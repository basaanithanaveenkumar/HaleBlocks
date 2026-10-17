"""Shared helpers of :mod:`hale_core.training.trainer` (Variant-agnostic training loop resolved from
registries).
"""

from __future__ import annotations

from typing import Any

from hale_core.config.run import RunConfig
from hale_core.distributed.ray.worker_context import WorkerContext
from hale_core.registry import get_trainer


def train(cfg: RunConfig, tokenizer=None, *, trainer: str = "default", **kwargs) -> Any:
    """Build the registered ``trainer`` and run it.

    With ``cfg.train.ray.enabled`` the run is launched on Ray Train instead: every Ray worker
    builds the trainer with ``parallel_strategy=ddp`` (unless ``fsdp`` is configured) and the
    driver gets a :class:`hale_core.distributed.RayRunResult` (metrics + checkpoint path)
    instead of the ``(model, tokenizer, losses)`` tuple, because the model lives in the workers.
    ``kwargs`` are shipped to the workers; pass ``<name>_factory=callable`` for objects that
    cannot be pickled - it is called inside each worker and passed as ``<name>``.
    """
    if cfg.train.ray.enabled and not WorkerContext.current().in_ray:
        from hale_core.training.ray_entry import launch_on_ray  # lazy: only needed with Ray

        return launch_on_ray(cfg, train, tokenizer, trainer=trainer, **kwargs)
    resolved: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key.endswith("_factory"):  # e.g. data_module_factory=... -> data_module=factory()
            resolved[key.removesuffix("_factory")] = value()
        else:
            resolved[key] = value
    return get_trainer(trainer)(cfg, tokenizer, **resolved).fit()
