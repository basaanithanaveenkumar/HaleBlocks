"""Distributed launch backends (Ray Train).

Everything here imports ``ray`` lazily, so ``import hale_core`` never requires Ray; install the
extra with ``pip install hale-blocks[ray]`` to use it.
"""

from hale_core.distributed.ray import RayLauncher, RayReporter, RayRunResult, WorkerContext

__all__ = ["RayLauncher", "RayReporter", "RayRunResult", "WorkerContext"]
