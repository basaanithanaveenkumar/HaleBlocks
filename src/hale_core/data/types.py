"""Public dataset type re-exports."""

from __future__ import annotations

from hale_core.data.kinds import DatasetKind, LLMDomain, TrainingStage, VLMModality
from hale_core.data.llm.types import (
    DEFAULT_SMOLLM2_RL_MIXTURE,
    DEFAULT_SMOLLM2_SFT_MIXTURE,
    SMOLLM2_PRETRAIN_STAGES,
    LLMDatasetSpec,
    LLMSample,
    StageMixture,
)
from hale_core.data.vlm.types import DEFAULT_VLM_MIXTURE, VLMDatasetSpec, VLMSample

Modality = VLMModality
DatasetSpec = VLMDatasetSpec

__all__ = [
    "DatasetKind",
    "TrainingStage",
    "LLMDomain",
    "VLMModality",
    "Modality",
    "VLMSample",
    "LLMSample",
    "VLMDatasetSpec",
    "LLMDatasetSpec",
    "DatasetSpec",
    "StageMixture",
    "DEFAULT_VLM_MIXTURE",
    "SMOLLM2_PRETRAIN_STAGES",
    "DEFAULT_SMOLLM2_SFT_MIXTURE",
    "DEFAULT_SMOLLM2_RL_MIXTURE",
]
