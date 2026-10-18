"""Rank information of the current process, whoever launched it."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from loguru import logger


@dataclass(frozen=True)
class WorkerContext:
    """Where this process sits in a (possibly) distributed run.

    Resolution order: a Ray Train worker (``ray.train.get_context()``), then an initialised
    ``torch.distributed`` process group (torchrun / manual), then the ``RANK`` / ``WORLD_SIZE``
    environment variables, then a single process.

    Attributes:
        rank: global rank in ``[0, world_size)``.
        local_rank: rank on this node (selects the GPU).
        world_size: number of processes.
        in_ray: whether this process is a Ray Train worker.
    """

    rank: int = 0
    local_rank: int = 0
    world_size: int = 1
    in_ray: bool = False

    @property
    def is_main(self) -> bool:
        """True on global rank 0 (the process that writes checkpoints and logs)."""
        return self.rank == 0

    @property
    def is_distributed(self) -> bool:
        """Whether more than one worker participates."""
        return self.world_size > 1

    @classmethod
    def current(cls) -> WorkerContext:
        """Context of the calling process (cheap; safe to call anywhere)."""
        ray_ctx = cls._from_ray()
        if ray_ctx is not None:
            return ray_ctx
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                return cls(
                    rank=dist.get_rank(),
                    local_rank=int(os.environ.get("LOCAL_RANK", 0)),
                    world_size=dist.get_world_size(),
                )
        except ImportError:  # pragma: no cover - torch is a hard dependency
            logger.debug("torch.distributed unavailable; reading rank from the environment")
        return cls(
            rank=int(os.environ.get("RANK", 0)),
            local_rank=int(os.environ.get("LOCAL_RANK", 0)),
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
        )

    @classmethod
    def _from_ray(cls) -> WorkerContext | None:
        # Only consult Ray if it is already imported: importing it just to find out we are not
        # inside a Ray worker would add seconds to every start-up.
        if "ray.train" not in sys.modules:
            return None
        import ray.train

        try:
            ctx = ray.train.get_context()
            return cls(
                rank=ctx.get_world_rank(),
                local_rank=ctx.get_local_rank(),
                world_size=ctx.get_world_size(),
                in_ray=True,
            )
        except RuntimeError:  # raised outside a function launched by TorchTrainer.fit()
            return None
