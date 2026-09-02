"""SmolLM2 pretraining — code datasets (Section 3.4)."""

from __future__ import annotations

from hale_core.data.kinds import LLMDomain, TrainingStage
from hale_core.data.llm.types import LLMDatasetSpec
from hale_core.registry import register_dataset


@register_dataset("starcoderdata")
def starcoderdata() -> LLMDatasetSpec:
    """StarCoderData: 250B tokens across 80 programming languages."""
    return LLMDatasetSpec(
        name="starcoderdata",
        hf_repo="bigcode/starcoderdata",
        stage=TrainingStage.PRETRAIN,
        domain=LLMDomain.CODE,
        split="train",
        text_fields=("content", "text", "code"),
        description="Filtered GitHub code corpus (10% in stage 1, 20% in stage 2).",
        smollm2_stage=1,
        mix_weight=0.10,
    )


@register_dataset("stack_edu")
def stack_edu() -> LLMDatasetSpec:
    """Stack-Edu: ~125B educational code tokens from StarCoder2Data."""
    return LLMDatasetSpec(
        name="stack_edu",
        hf_repo="HuggingFaceTB/stack-edu",
        stage=TrainingStage.PRETRAIN,
        domain=LLMDomain.CODE,
        split="train",
        text_fields=("content", "text", "code"),
        description="Classifier-filtered educational code (16% stage 3, 24% stage 4).",
        smollm2_stage=3,
        mix_weight=0.16,
    )


@register_dataset("starcoder2data")
def starcoder2data() -> LLMDatasetSpec:
    """StarCoder2Data: 900B tokens across 600+ languages."""
    return LLMDatasetSpec(
        name="starcoder2data",
        hf_repo="bigcode/starcoder2data",
        stage=TrainingStage.PRETRAIN,
        domain=LLMDomain.CODE,
        split="train",
        text_fields=("content", "text", "code"),
        description="Full StarCoder2 corpus; used for low-resource languages in stage 3.",
        smollm2_stage=3,
    )


@register_dataset("starcoder2_jupyter")
def starcoder2_jupyter() -> LLMDatasetSpec:
    """StarCoder2 Jupyter notebooks: code interleaved with explanations."""
    return LLMDatasetSpec(
        name="starcoder2_jupyter",
        hf_repo="bigcode/starcoder2data",
        stage=TrainingStage.PRETRAIN,
        domain=LLMDomain.CODE,
        hf_subset="jupyter",
        split="train",
        text_fields=("content", "text", "code"),
        description="Jupyter notebooks from StarCoder2 added in stage 3.",
        smollm2_stage=3,
    )
