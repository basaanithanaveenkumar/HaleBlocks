"""``TrainingStage``: TrainingStage (Shared dataset kind and stage enums)."""

from __future__ import annotations

from enum import StrEnum


class TrainingStage(StrEnum):
    PRETRAIN = "pretrain"
    FINETUNE = "finetune"
    RL = "rl"
