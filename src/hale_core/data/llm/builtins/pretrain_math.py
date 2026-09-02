"""SmolLM2 pretraining — math datasets (Section 3.3)."""

from __future__ import annotations

from hale_core.data.kinds import LLMDomain, TrainingStage
from hale_core.data.llm.types import LLMDatasetSpec
from hale_core.registry import register_dataset


@register_dataset("openwebmath")
def openwebmath() -> LLMDatasetSpec:
    """OpenWebMath: 12B math tokens from Common Crawl."""
    return LLMDatasetSpec(
        name="openwebmath",
        hf_repo="open-web-math/open-web-math",
        stage=TrainingStage.PRETRAIN,
        domain=LLMDomain.MATH,
        split="train",
        text_fields=("text", "content"),
        description="Math-specific web crawl with LaTeX preservation.",
        smollm2_stage=2,
        mix_weight=0.05,
    )


@register_dataset("infimm_webmath")
def infimm_webmath() -> LLMDatasetSpec:
    """InfiMM-WebMath: 40B text tokens for mathematical reasoning."""
    return LLMDatasetSpec(
        name="infimm_webmath",
        hf_repo="BAAI/Infinity-MM",
        stage=TrainingStage.PRETRAIN,
        domain=LLMDomain.MATH,
        hf_subset="WebMath",
        split="train",
        text_fields=("text", "content"),
        description="Large-scale math web text from InfiMM-WebMath project.",
        smollm2_stage=3,
        mix_weight=0.05,
    )


@register_dataset("finemath4_plus")
def finemath4_plus() -> LLMDatasetSpec:
    """FineMath4+: 10B tokens, scores 4-5 (step-by-step reasoning)."""
    return LLMDatasetSpec(
        name="finemath4_plus",
        hf_repo="HuggingFaceTB/finemath",
        stage=TrainingStage.PRETRAIN,
        domain=LLMDomain.MATH,
        hf_subset="finemath-4plus",
        split="train",
        text_fields=("text", "content"),
        description="Highest-quality FineMath subset for annealing (stage 4).",
        smollm2_stage=4,
        mix_weight=0.10,
    )


@register_dataset("finemath3_plus")
def finemath3_plus() -> LLMDatasetSpec:
    """FineMath3+: 34B tokens, scores 3-5."""
    return LLMDatasetSpec(
        name="finemath3_plus",
        hf_repo="HuggingFaceTB/finemath",
        stage=TrainingStage.PRETRAIN,
        domain=LLMDomain.MATH,
        hf_subset="finemath-3plus",
        split="train",
        text_fields=("text", "content"),
        description="Broader FineMath subset including score-3 content.",
        smollm2_stage=4,
    )


@register_dataset("infi_webmath4_plus")
def infi_webmath4_plus() -> LLMDatasetSpec:
    """Infi-WebMath4+: 8.5B tokens from InfiMM re-filtered."""
    return LLMDatasetSpec(
        name="infi_webmath4_plus",
        hf_repo="HuggingFaceTB/finemath",
        stage=TrainingStage.PRETRAIN,
        domain=LLMDomain.MATH,
        hf_subset="infiwebmath-4plus",
        split="train",
        text_fields=("text", "content"),
        description="Re-filtered InfiMM-WebMath high-quality subset.",
        smollm2_stage=4,
    )


@register_dataset("infi_webmath3_plus")
def infi_webmath3_plus() -> LLMDatasetSpec:
    """Infi-WebMath3+: 20.5B tokens from InfiMM re-filtered."""
    return LLMDatasetSpec(
        name="infi_webmath3_plus",
        hf_repo="HuggingFaceTB/finemath",
        stage=TrainingStage.PRETRAIN,
        domain=LLMDomain.MATH,
        hf_subset="infiwebmath-3plus",
        split="train",
        text_fields=("text", "content"),
        description="Broader re-filtered InfiMM-WebMath subset.",
        smollm2_stage=4,
        mix_weight=0.03,
    )


@register_dataset("aug_gsm8k")
def aug_gsm8k() -> LLMDatasetSpec:
    """AugGSM8K: augmented GSM8K training set for math annealing."""
    return LLMDatasetSpec(
        name="aug_gsm8k",
        hf_repo="meta-math/AugGSM8K",
        stage=TrainingStage.PRETRAIN,
        domain=LLMDomain.MATH,
        split="train",
        text_fields=("text", "question", "answer"),
        prompt_fields=("question",),
        response_fields=("answer",),
        description="Augmented grade-school math problems (0.02% in stage 4).",
        smollm2_stage=4,
        mix_weight=0.0002,
    )
