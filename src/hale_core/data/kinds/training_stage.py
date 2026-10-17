"""``TrainingStage``: Training stage a dataset belongs to (pre-training, SFT, preference …)."""

from __future__ import annotations

from enum import StrEnum


class TrainingStage(StrEnum):
    """Training stage a dataset belongs to (pre-training, SFT, preference …)."""

    PRETRAIN = "pretrain"
    FINETUNE = "finetune"
    RL = "rl"
