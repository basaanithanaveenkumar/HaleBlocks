"""Driver-side launcher: run a training function on a Ray cluster with ``TorchTrainer``."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from loguru import logger

from hale_core.config.sections.ray import RayConfig
from hale_core.distributed.ray.dependency import require_ray
from hale_core.distributed.ray.run_result import RayRunResult

TrainFunction = Callable[[dict[str, Any]], Any]


class RayLauncher:
    """Translate a :class:`RayConfig` into a Ray Train run.

    ``launch(train_fn, loop_config)`` starts (or joins) Ray, creates ``config.num_workers``
    workers with ``torch.distributed`` initialised (``RANK`` / ``LOCAL_RANK`` / ``WORLD_SIZE``
    set, NCCL on GPU and Gloo on CPU), calls ``train_fn(loop_config)`` in every worker and
    returns a :class:`RayRunResult`. ``train_fn`` and ``loop_config`` are shipped to the workers
    with cloudpickle, so they must be importable / picklable (pass factories for objects that
    are not).

    Example::

        def train_fn(loop_config):
            ctx = WorkerContext.current()          # rank, world_size, ...
            ...                                    # normal DDP training loop
            RayReporter().report({"loss": loss}, checkpoint_dir="checkpoints/")

        result = RayLauncher(cfg.train.ray).launch(train_fn, {"lr": 1e-3})
        print(result.metrics, result.checkpoint_path)

    Args:
        config: the ``train.ray`` section.
    """

    def __init__(self, config: RayConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------ public API
    def launch(
        self,
        train_fn: TrainFunction,
        loop_config: dict[str, Any] | None = None,
        *,
        raise_on_error: bool = True,
    ) -> RayRunResult:
        """Run ``train_fn(loop_config)`` on every worker and wait for completion."""
        ray, _, _ = require_ray()
        started_here = self._ensure_initialised(ray)
        try:
            trainer = self.build_trainer(train_fn, loop_config or {})
            logger.info(
                "ray: launching {} worker(s) use_gpu={} backend={} storage={}",
                self.config.num_workers,
                self.config.use_gpu,
                self.config.resolved_backend(),
                self.storage_path,
            )
            try:
                result = trainer.fit()
            except Exception as exc:  # Ray raises TrainingFailedError on worker failure
                if raise_on_error:
                    raise
                logger.exception("ray: training failed")
                return RayRunResult(storage_path=self.storage_path, error=exc)
            return RayRunResult(
                metrics=dict(result.metrics or {}),
                checkpoint_path=None if result.checkpoint is None else result.checkpoint.path,
                storage_path=self.storage_path,
                error=result.error,
            )
        finally:
            if started_here:
                ray.shutdown()

    def build_trainer(self, train_fn: TrainFunction, loop_config: dict[str, Any]):
        """The ``TorchTrainer`` for this config (exposed for inspection and tests)."""
        _, train, train_torch = require_ray()
        cfg = self.config
        scaling = train.ScalingConfig(
            num_workers=cfg.num_workers,
            use_gpu=cfg.use_gpu,
            resources_per_worker=dict(cfg.resources_per_worker) or None,
            placement_strategy=cfg.placement_strategy,
        )
        run = train.RunConfig(
            name=cfg.run_name,
            storage_path=self.storage_path,
            failure_config=train.FailureConfig(max_failures=cfg.max_failures),
            checkpoint_config=train.CheckpointConfig(num_to_keep=cfg.num_to_keep),
        )
        return train_torch.TorchTrainer(
            train_loop_per_worker=train_fn,
            train_loop_config=loop_config,
            torch_config=train_torch.TorchConfig(
                backend=cfg.resolved_backend(), timeout_s=cfg.timeout_s
            ),
            scaling_config=scaling,
            run_config=run,
        )

    @property
    def storage_path(self) -> str:
        """Absolute storage path (Ray requires absolute local paths); URIs pass through."""
        path = self.config.storage_path
        return path if "://" in path else str(Path(path).expanduser().resolve())

    # ------------------------------------------------------------------ helpers
    def _ensure_initialised(self, ray) -> bool:
        """Start / join Ray unless already connected. Returns True if this call started it."""
        if ray.is_initialized():
            return False
        kwargs: dict[str, Any] = dict(self.config.init_kwargs)
        if self.config.runtime_env:
            kwargs["runtime_env"] = dict(self.config.runtime_env)
        if self.config.address is None:
            # Local instance: no dashboard (Ray rejects this flag when joining a cluster).
            kwargs.setdefault("include_dashboard", False)
        ray.init(address=self.config.address, **kwargs)
        return True
