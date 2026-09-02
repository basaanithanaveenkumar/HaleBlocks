"""SmolLM2 preference learning / RL datasets (Section 5.3)."""

from __future__ import annotations

from hale_core.data.kinds import LLMDomain, TrainingStage
from hale_core.data.llm.types import LLMDatasetSpec
from hale_core.registry import register_dataset


@register_dataset("ultrafeedback")
def ultrafeedback() -> LLMDatasetSpec:
    """UltraFeedback: chosen DPO dataset for SmolLM2 alignment."""
    return LLMDatasetSpec(
        name="ultrafeedback",
        hf_repo="HuggingFaceH4/ultrafeedback_binarized",
        stage=TrainingStage.RL,
        domain=LLMDomain.PREFERENCE,
        split="train_prefs",
        prompt_fields=("prompt", "instruction"),
        chosen_fields=("chosen", "chosen_response"),
        rejected_fields=("rejected", "rejected_response"),
        conversation_fields=("messages",),
        description="Primary DPO dataset; best on MT-Bench, MMLU-Pro, MATH.",
    )


@register_dataset("ultrainteract")
def ultrainteract() -> LLMDatasetSpec:
    """UltraInteract: preference trees for reasoning generalization."""
    return LLMDatasetSpec(
        name="ultrainteract",
        hf_repo="stabilityai/UltraInteract",
        stage=TrainingStage.RL,
        domain=LLMDomain.PREFERENCE,
        split="train",
        prompt_fields=("prompt", "instruction"),
        chosen_fields=("chosen",),
        rejected_fields=("rejected",),
        description="Preference tree data for LLM reasoning alignment.",
    )


@register_dataset("capybara")
def capybara() -> LLMDatasetSpec:
    """Capybara: diverse multi-turn synthetic preference data."""
    return LLMDatasetSpec(
        name="capybara",
        hf_repo="LDJnr/Capybara",
        stage=TrainingStage.RL,
        domain=LLMDomain.PREFERENCE,
        split="train",
        conversation_fields=("messages", "conversations"),
        chosen_fields=("chosen",),
        rejected_fields=("rejected",),
        description="Synthetic diverse conversations for preference learning.",
    )


@register_dataset("orca_dpo")
def orca_dpo() -> LLMDatasetSpec:
    """ORCA DPO pairs from Intel for preference optimization."""
    return LLMDatasetSpec(
        name="orca_dpo",
        hf_repo="Intel/orca_dpo_pairs",
        stage=TrainingStage.RL,
        domain=LLMDomain.PREFERENCE,
        split="train",
        prompt_fields=("question", "prompt", "instruction"),
        chosen_fields=("chosen", "chosen_response"),
        rejected_fields=("rejected", "rejected_response"),
        description="Intel ORCA preference pairs for DPO training.",
    )
