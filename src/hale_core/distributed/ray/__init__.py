"""Ray Train backend: launch any training function on N workers with torch.distributed wired up.

* :class:`RayLauncher`  - driver: ``RayConfig`` -> ``TorchTrainer.fit()`` -> :class:`RayRunResult`
* :class:`WorkerContext` - rank / world size of this process (Ray, torchrun or single process)
* :class:`RayReporter`  - worker side: report metrics and checkpoints to Ray (no-op outside Ray)
"""

from hale_core.distributed.ray.launcher import RayLauncher
from hale_core.distributed.ray.reporter import RayReporter
from hale_core.distributed.ray.run_result import RayRunResult
from hale_core.distributed.ray.worker_context import WorkerContext

__all__ = ["RayLauncher", "RayReporter", "RayRunResult", "WorkerContext"]
