"""Ray Train sub-config: run training on a Ray cluster (or a local Ray instance)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from hale_core.config.sections.common import StrictModel


class RayConfig(StrictModel):
    """How to launch training with Ray Train (``ray.train.torch.TorchTrainer``).

    Ray starts ``num_workers`` training processes (one per GPU, or CPU processes), wires
    ``torch.distributed`` between them and restarts them on failure. Inside every worker the
    normal training code runs with ``parallel_strategy=ddp`` (or ``fsdp``), so nothing else in
    the training loop changes. Disabled by default: ``enabled: false`` keeps the plain
    single-process / torchrun behaviour.

    YAML example::

        train:
          ray:
            enabled: true
            address: auto          # join a running cluster; omit to start a local Ray instance
            num_workers: 8
            use_gpu: true
            resources_per_worker: {CPU: 8, GPU: 1}
            storage_path: /shared/ray_results
            max_failures: 3
    """

    enabled: bool = False
    """Launch training through Ray Train (needs ``pip install hale-blocks[ray]``)."""

    # ── Cluster ───────────────────────────────────────────────────────────────
    address: str | None = None
    """``None`` starts a local Ray instance; ``"auto"`` / ``"ray://host:10001"`` joins a cluster."""
    init_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Extra ``ray.init`` keyword arguments (e.g. ``num_cpus`` for a local instance)."""
    runtime_env: dict[str, Any] = Field(default_factory=dict)
    """Ray runtime environment for every worker (``pip``, ``env_vars``, ``working_dir`` ...)."""

    # ── Scaling ───────────────────────────────────────────────────────────────
    num_workers: int = Field(default=1, ge=1)
    """Number of training processes (data-parallel replicas)."""
    use_gpu: bool = False
    """Give each worker one GPU (``resources_per_worker`` can request more / fractional)."""
    resources_per_worker: dict[str, float] = Field(default_factory=dict)
    """Ray resources per worker, e.g. ``{"CPU": 4, "GPU": 1}`` (empty = 1 CPU [+ 1 GPU])."""
    placement_strategy: Literal["PACK", "SPREAD", "STRICT_PACK", "STRICT_SPREAD"] = "PACK"
    """How workers are spread over nodes (``PACK`` keeps them on as few nodes as possible)."""

    # ── torch.distributed inside the workers ─────────────────────────────────
    backend: Literal["auto", "nccl", "gloo"] = "auto"
    """Process-group backend; ``auto`` = NCCL with GPUs, Gloo on CPU."""
    timeout_s: int = Field(default=1800, ge=1)
    """Process-group initialisation / collective timeout in seconds."""

    # ── Results, checkpoints, fault tolerance ────────────────────────────────
    storage_path: str = "ray_results"
    """Where Ray persists reported checkpoints and results (local path, NFS or ``s3://``)."""
    run_name: str | None = None
    """Name of the run directory under ``storage_path`` (default: Ray generates one)."""
    max_failures: int = Field(default=0, ge=-1)
    """Worker-group restarts before giving up (``-1`` = unlimited)."""
    num_to_keep: int | None = Field(default=2, ge=1)
    """Reported checkpoints to keep (``None`` = keep all)."""

    def resolved_backend(self) -> str:
        """The concrete process-group backend for this config."""
        if self.backend != "auto":
            return self.backend
        return "nccl" if self.use_gpu or self.resources_per_worker.get("GPU", 0) > 0 else "gloo"
