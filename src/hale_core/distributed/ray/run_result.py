"""What a Ray-launched training run returns to the driver."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class RayRunResult:
    """Outcome of :meth:`RayLauncher.launch`.

    The trained model lives in the worker processes, so the driver gets the last reported
    metrics and the path of the last reported checkpoint (load it with the repo's normal
    checkpoint loader).

    Attributes:
        metrics: metrics of the last report that carried a checkpoint (Ray Train V2 keeps
            metrics together with checkpoints only; empty if no checkpoint was reported).
        checkpoint_path: directory of the last reported checkpoint, or ``None``.
        storage_path: run directory holding all results and checkpoints.
        error: the worker exception if the run failed and ``raise_on_error`` was false.
    """

    metrics: dict[str, Any] = field(default_factory=dict)
    checkpoint_path: str | None = None
    storage_path: str | None = None
    error: BaseException | None = None

    @property
    def ok(self) -> bool:
        """Whether the Ray run finished without an error."""
        return self.error is None
