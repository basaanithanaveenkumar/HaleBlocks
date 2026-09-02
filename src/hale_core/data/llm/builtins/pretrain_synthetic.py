"""SmolLM2 pretraining — synthetic and long-context datasets."""

from __future__ import annotations

from hale_core.data.kinds import LLMDomain, TrainingStage
from hale_core.data.llm.types import LLMDatasetSpec
from hale_core.registry import register_dataset


@register_dataset("cosmopedia_v2")
def cosmopedia_v2() -> LLMDatasetSpec:
    """Cosmopedia v2: 30B synthetic textbooks, blogs, and stories."""
    return LLMDatasetSpec(
        name="cosmopedia_v2",
        hf_repo="HuggingFaceTB/cosmopedia-v2",
        stage=TrainingStage.PRETRAIN,
        domain=LLMDomain.SYNTHETIC,
        split="train",
        text_fields=("text", "content", "prompt"),
        description="Synthetic educational content (4% in stage 4 annealing).",
        smollm2_stage=4,
        mix_weight=0.04,
    )


@register_dataset("dolma_books")
def dolma_books() -> LLMDatasetSpec:
    """Dolma books subset for 8k context extension."""
    return LLMDatasetSpec(
        name="dolma_books",
        hf_repo="allenai/dolma",
        stage=TrainingStage.PRETRAIN,
        domain=LLMDomain.WEB,
        hf_subset="books",
        split="train",
        text_fields=("text", "content"),
        description="Book corpus from Dolma for long-context extension (20% of CLE mix).",
    )
