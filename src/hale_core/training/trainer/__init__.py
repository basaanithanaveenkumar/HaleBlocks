"""Variant-agnostic training loop resolved from registries.

Pluggable training modules
--------------------------
Three optional modules are auto-constructed from cfg.train and injected into
the training loop:

  AMPManager      (cfg.train.amp)     — mixed precision fp16/bf16
  PruningManager  (cfg.train.pruning) — in-training weight pruning
  QATManager      (cfg.train.quant)   — quantization-aware training

After training, ExportManager is called when cfg.train.export.enabled is True.

All modules are disabled (no-op) by default; enable each independently via YAML:

    train:
      amp:
        enabled: true
        dtype: bf16
      pruning:
        enabled: true
        target_sparsity: 0.5
      quant:
        enabled: true
        qat_start_step: 5000
      export:
        enabled: true
        targets: [onnx, mlx]
"""

from hale_core.training.trainer._helpers import train
from hale_core.training.trainer.data_module import DataModule
from hale_core.training.trainer.evaluator import Evaluator
from hale_core.training.trainer.trainer import Trainer

__all__ = ["DataModule", "Evaluator", "Trainer", "train"]
