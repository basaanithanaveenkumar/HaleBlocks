"""SmolLM2 pretraining — English web datasets (Section 3.2)."""

from __future__ import annotations

from hale_core.data.kinds import LLMDomain, TrainingStage
from hale_core.data.llm.types import LLMDatasetSpec
from hale_core.registry import register_dataset


@register_dataset("fineweb_edu")
def fineweb_edu() -> LLMDatasetSpec:
    """FineWeb-Edu: 1.3T educational web tokens (60% of web mix in stage 1)."""
    return LLMDatasetSpec(
        name="fineweb_edu",
        hf_repo="HuggingFaceFW/fineweb-edu",
        stage=TrainingStage.PRETRAIN,
        domain=LLMDomain.WEB,
        split="train",
        text_fields=("text", "content"),
        description="Classifier-filtered educational web text from Common Crawl.",
        smollm2_stage=1,
        mix_weight=0.60,
    )


@register_dataset("dclm")
def dclm() -> LLMDatasetSpec:
    """DCLM baseline: 3.8T filtered web tokens (40% of web mix in stage 1)."""
    return LLMDatasetSpec(
        name="dclm",
        hf_repo="mlfoundations/dclm-baseline-1.0",
        stage=TrainingStage.PRETRAIN,
        domain=LLMDomain.WEB,
        hf_subset="default",
        split="train",
        text_fields=("text", "content"),
        description="DataComp-LM baseline corpus with fastText filtering.",
        smollm2_stage=1,
        mix_weight=0.40,
    )
