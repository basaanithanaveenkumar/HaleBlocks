"""Worker-side reporting of metrics and checkpoints to Ray Train."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hale_core.distributed.ray.worker_context import WorkerContext


class RayReporter:
    """Report metrics (and optionally a checkpoint directory) from inside a training loop.

    Outside a Ray Train worker every call is a no-op, so training code can report
    unconditionally. Inside Ray, **every rank must call** :meth:`report` at the same points
    (Ray synchronises the reports); only rank 0 attaches the checkpoint, which Ray then copies
    to ``RayConfig.storage_path`` and uses to restart the run after a failure. Ray Train V2
    surfaces metrics in the driver's result only for reports that carry a checkpoint.
    """

    def __init__(self) -> None:
        self.context = WorkerContext.current()

    @property
    def active(self) -> bool:
        """Whether this process runs inside a Ray Train worker (metrics are forwarded only then)."""
        return self.context.in_ray

    def report(self, metrics: Mapping[str, Any], checkpoint_dir: str | Path | None = None) -> None:
        """Send ``metrics`` (numbers) and, on rank 0, the checkpoint in ``checkpoint_dir``."""
        if not self.active:
            return
        import ray.train

        clean = {k: float(v) if hasattr(v, "__float__") else v for k, v in metrics.items()}
        checkpoint = None
        if checkpoint_dir is not None and self.context.is_main and Path(checkpoint_dir).is_dir():
            checkpoint = ray.train.Checkpoint.from_directory(str(checkpoint_dir))
        ray.train.report(clean, checkpoint=checkpoint)

    @staticmethod
    def latest_checkpoint_dir() -> str | None:
        """Directory of the checkpoint Ray restored this worker group from (after a failure)."""
        import sys

        if "ray.train" not in sys.modules:
            return None
        import ray.train

        try:
            checkpoint = ray.train.get_checkpoint()
        except RuntimeError:
            return None
        return None if checkpoint is None else checkpoint.path
