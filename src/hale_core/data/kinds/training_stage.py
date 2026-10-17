"""``TrainingStage``: TrainingStage (Shared dataset kind and stage enums)."""

from __future__ import annotations

from enum import StrEnum


class TrainingStage(StrEnum):
    """Training stage a dataset belongs to (pre-training, SFT, preference …)."""
    PRETRAIN = "pretrain"
    FINETUNE = "finetune"
    RL = "rl"
