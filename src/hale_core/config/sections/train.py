from __future__ import annotations

from typing import Literal

from pydantic import Field

from hale_core.config.sections.amp import AMPConfig
from hale_core.config.sections.common import StrictModel
from hale_core.config.sections.ray import RayConfig
from hale_core.config.sections.pruning import PruningConfig
from hale_core.config.sections.quant import ExportConfig, QuantConfig
from hale_core.config.sections.schedule import SchedulerConfig


class TrainConfig(StrictModel):
    """Training loop hyper-parameters.

    ``scheduler`` is a nested sub-config; it defaults to a constant LR
    (no warmup, no decay) which matches the pre-existing behaviour.

    YAML example::

        train:
          steps: 10000
          lr: 3e-4
          optimizer_type: adamw
          scheduler:
            type: warmup_cosine
            warmup_steps: 200
            min_lr_ratio: 0.1
    """

    # ── Steps / epochs ─────────────────────────────────────────────────────────
    steps: int | None = 300
    epochs: int | None = None

    # ── Optimisation ──────────────────────────────────────────────────────────
    lr: float = 1e-3
    batch_size: int = 8
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    optimizer_type: str = "adamw"

    # ── LR schedule ───────────────────────────────────────────────────────────
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

    # ── Loss ──────────────────────────────────────────────────────────────────
    loss_type: Literal["ce", "focal", "label_smoothing"] = "ce"
    focal_gamma: float = 2.0
    focal_alpha: float = 1.0

    # ── Logging / checkpointing ────────────────────────────────────────────────
    log_every: int = 50
    seed: int = 0
    checkpoint_path: str = "checkpoints/last.pt"
    checkpoint_every_epoch: bool = True
    resume: bool = True

    # ── Distributed ───────────────────────────────────────────────────────────
    parallel_strategy: Literal["none", "dp", "ddp", "fsdp"] = "none"
    distributed_backend: str | None = None
    find_unused_parameters: bool = False
    gradient_as_bucket_view: bool = True
    static_graph: bool = False
    ray: RayConfig = Field(default_factory=RayConfig)
    """Launch through Ray Train (multi-GPU / multi-node, fault tolerant). Disabled by default."""

    # ── Mixed precision ────────────────────────────────────────────────────────
    amp: AMPConfig = Field(default_factory=AMPConfig)
    """Automatic Mixed Precision (fp16/bf16).  Disabled by default (enabled=False).
    Set amp.enabled=True and amp.dtype='bf16' for Ampere+ GPUs."""

    # ── In-training pruning ────────────────────────────────────────────────────
    pruning: PruningConfig = Field(default_factory=PruningConfig)
    """Weight pruning during training.  Disabled by default (enabled=False).
    Set pruning.enabled=True, pruning.target_sparsity=0.5 for 50% sparse."""

    # ── Quantization-aware training ───────────────────────────────────────────
    quant: QuantConfig = Field(default_factory=QuantConfig)
    """QAT with fake-quantization nodes.  Disabled by default (enabled=False).
    Set quant.enabled=True to insert int8 fake-quant nodes during training."""

    # ── Export ────────────────────────────────────────────────────────────────
    export: ExportConfig = Field(default_factory=ExportConfig)
    """Post-training export to ONNX, TensorRT, MLX.  Disabled by default.
    Set export.enabled=True and export.targets=['onnx','tensorrt'] to export."""
