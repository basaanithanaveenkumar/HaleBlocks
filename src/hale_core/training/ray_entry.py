"""Run a repo's ``train(cfg, ...)`` function on Ray Train workers (``train.ray.enabled: true``).

Shared by every Hale* repo: ``hale_core.training.train`` and the repos' own ``train`` functions
all call :func:`launch_on_ray` with themselves as ``train_fn``. The config only needs the
duck-typed fields ``train.ray`` (:class:`RayConfig`), ``train.parallel_strategy`` and
``train.checkpoint_path``.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from hale_core.distributed.ray.launcher import RayLauncher
from hale_core.distributed.ray.reporter import RayReporter
from hale_core.distributed.ray.run_result import RayRunResult
from hale_core.distributed.ray.worker_context import WorkerContext


def launch_on_ray(
    cfg: Any, train_fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> RayRunResult:
    """Driver side: run ``train_fn(cfg, *args, **kwargs)`` on ``cfg.train.ray.num_workers`` workers.

    ``train_fn`` must be importable in the workers (a module-level function of an installed
    package). The config travels as its class + ``model_dump()`` so repo-specific config
    classes survive; ``args`` / ``kwargs`` are cloudpickled by Ray.
    """
    loop_config = {
        "config_cls": type(cfg),
        "config": cfg.model_dump(),
        "train_fn": train_fn,
        "args": args,
        "kwargs": kwargs,
    }
    result = RayLauncher(cfg.train.ray).launch(ray_train_worker, loop_config)
    logger.info("ray: finished metrics={} checkpoint={}", result.metrics, result.checkpoint_path)
    return result


def ray_train_worker(loop_config: dict[str, Any]) -> None:
    """Worker side: rebuild the config, restore Ray's checkpoint if any, call ``train_fn``."""
    cfg = loop_config["config_cls"].model_validate(loop_config["config"])
    # Ray has already created the process group; the trainer just has to use it.
    if cfg.train.parallel_strategy in ("none", "dp"):
        cfg.train.parallel_strategy = "ddp"
    restore_ray_checkpoint(Path(cfg.train.checkpoint_path).parent)
    ctx = WorkerContext.current()
    logger.info("ray worker rank={}/{} local_rank={}", ctx.rank, ctx.world_size, ctx.local_rank)
    loop_config["train_fn"](cfg, *loop_config["args"], **loop_config["kwargs"])


def restore_ray_checkpoint(checkpoint_dir: Path) -> None:
    """After a worker failure Ray restarts the group with the last reported checkpoint; copy it
    to where the trainer looks for checkpoints so ``train.resume`` continues from it."""
    restored = RayReporter.latest_checkpoint_dir()
    if restored is None:
        return
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copytree(restored, checkpoint_dir, dirs_exist_ok=True)
    logger.info("ray: restored checkpoint {} -> {}", restored, checkpoint_dir)
