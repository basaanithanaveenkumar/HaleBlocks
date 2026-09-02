"""Shared dataset kind and stage enums."""

from __future__ import annotations

from enum import StrEnum


class DatasetKind(StrEnum):
    LLM = "llm"
    VLM = "vlm"


class TrainingStage(StrEnum):
    PRETRAIN = "pretrain"
    FINETUNE = "finetune"
    RL = "rl"


class LLMDomain(StrEnum):
    WEB = "web"
    MATH = "math"
    CODE = "code"
    SYNTHETIC = "synthetic"
    INSTRUCTION = "instruction"
    PREFERENCE = "preference"


class VLMModality(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    MULTI_IMAGE = "multi_image"
    VIDEO = "video"
    MIXED = "mixed"
