"""SmolLM2 supervised fine-tuning datasets (Section 5 / SmolTalk)."""

from __future__ import annotations

from hale_core.data.kinds import LLMDomain, TrainingStage
from hale_core.data.llm.types import LLMDatasetSpec
from hale_core.registry import register_dataset


@register_dataset("smoltalk")
def smoltalk() -> LLMDatasetSpec:
    """SmolTalk: 1.1M instruction-response pairs (full SFT mixture)."""
    return LLMDatasetSpec(
        name="smoltalk",
        hf_repo="HuggingFaceTB/smoltalk",
        stage=TrainingStage.FINETUNE,
        domain=LLMDomain.INSTRUCTION,
        split="train",
        conversation_fields=("messages", "conversations"),
        description="Complete SmolTalk SFT dataset for SmolLM2-Instruct.",
    )


@register_dataset("smol_smoltalk")
def smol_smoltalk() -> LLMDatasetSpec:
    """Smol-SmolTalk: filtered SmolTalk for 135M/360M models."""
    return LLMDatasetSpec(
        name="smol_smoltalk",
        hf_repo="HuggingFaceTB/smol-smoltalk",
        stage=TrainingStage.FINETUNE,
        domain=LLMDomain.INSTRUCTION,
        split="train",
        conversation_fields=("messages", "conversations"),
        description="Capacity-filtered SmolTalk for smaller SmolLM2 variants.",
    )


@register_dataset("magpie_ultra")
def magpie_ultra() -> LLMDatasetSpec:
    """MagPie-Ultra: 1M three-turn conversations (431k in SmolTalk)."""
    return LLMDatasetSpec(
        name="magpie_ultra",
        hf_repo="HuggingFaceTB/smoltalk",
        stage=TrainingStage.FINETUNE,
        domain=LLMDomain.INSTRUCTION,
        split="train",
        source_filter="magpie_ultra",
        conversation_fields=("messages", "conversations"),
        description="Multi-turn synthetic conversations from Llama-3.1-405B.",
    )


@register_dataset("smol_constraint")
def smol_constraint() -> LLMDatasetSpec:
    """Smol-Constraint: 36k IFEval-style constrained instructions."""
    return LLMDatasetSpec(
        name="smol_constraint",
        hf_repo="HuggingFaceTB/smoltalk",
        stage=TrainingStage.FINETUNE,
        domain=LLMDomain.INSTRUCTION,
        split="train",
        source_filter="smol_constraint",
        prompt_fields=("instruction", "prompt"),
        response_fields=("response", "answer"),
        description="Detailed constraint-following instructions.",
    )


@register_dataset("smol_rewrite")
def smol_rewrite() -> LLMDatasetSpec:
    """Smol-Rewrite: 56k text rewriting instruction pairs."""
    return LLMDatasetSpec(
        name="smol_rewrite",
        hf_repo="HuggingFaceTB/smoltalk",
        stage=TrainingStage.FINETUNE,
        domain=LLMDomain.INSTRUCTION,
        split="train",
        source_filter="smol_rewrite",
        prompt_fields=("instruction", "prompt"),
        response_fields=("response", "answer"),
        description="Text rewriting tasks for OpenRewrite-Eval performance.",
    )


@register_dataset("smol_summarization")
def smol_summarization() -> LLMDatasetSpec:
    """Smol-Summarization: 101k summarization instruction pairs."""
    return LLMDatasetSpec(
        name="smol_summarization",
        hf_repo="HuggingFaceTB/smoltalk",
        stage=TrainingStage.FINETUNE,
        domain=LLMDomain.INSTRUCTION,
        split="train",
        source_filter="smol_summarization",
        prompt_fields=("instruction", "prompt"),
        response_fields=("response", "answer"),
        description="Summarization tasks from diverse synthetic source texts.",
    )


@register_dataset("numinamath_cot")
def numinamath_cot() -> LLMDatasetSpec:
    """NuminaMath-CoT: chain-of-thought math instructions (112k in SmolTalk)."""
    return LLMDatasetSpec(
        name="numinamath_cot",
        hf_repo="AI-MO/NuminaMath-CoT",
        stage=TrainingStage.FINETUNE,
        domain=LLMDomain.MATH,
        split="train",
        prompt_fields=("problem", "question", "instruction"),
        response_fields=("solution", "answer", "response"),
        description="CoT math reasoning for MATH benchmark improvement.",
    )


@register_dataset("metamathqa")
def metamathqa() -> LLMDatasetSpec:
    """MetaMathQA: bootstrapped math QA (50k in SmolTalk)."""
    return LLMDatasetSpec(
        name="metamathqa",
        hf_repo="meta-math/MetaMathQA",
        stage=TrainingStage.FINETUNE,
        domain=LLMDomain.MATH,
        split="train",
        prompt_fields=("query", "question", "instruction"),
        response_fields=("response", "answer"),
        description="Augmented math QA for GSM8K performance.",
    )


@register_dataset("self_oss_starcoder2_instruct")
def self_oss_starcoder2_instruct() -> LLMDatasetSpec:
    """Self-OSS-StarCoder2-Instruct: 50k Python instruction pairs."""
    return LLMDatasetSpec(
        name="self_oss_starcoder2_instruct",
        hf_repo="bigcode/self-oss-instruct-sc2",
        stage=TrainingStage.FINETUNE,
        domain=LLMDomain.CODE,
        split="train",
        prompt_fields=("instruction", "prompt"),
        response_fields=("response", "answer", "code"),
        description="High-quality Python code instruction data.",
    )


@register_dataset("apigen_function_calling")
def apigen_function_calling() -> LLMDatasetSpec:
    """APIGen-Function-Calling: 87.5k verifiable function-calling samples."""
    return LLMDatasetSpec(
        name="apigen_function_calling",
        hf_repo="Salesforce/xlam-function-calling-60k",
        stage=TrainingStage.FINETUNE,
        domain=LLMDomain.INSTRUCTION,
        split="train",
        conversation_fields=("messages", "conversations"),
        description="Function calling instructions with verifiable outputs.",
    )


@register_dataset("systemchats2")
def systemchats2() -> LLMDatasetSpec:
    """SystemChats 2.0: 30k system-prompt conversation samples."""
    return LLMDatasetSpec(
        name="systemchats2",
        hf_repo="cognitivecomputations/SystemChat-2.0",
        stage=TrainingStage.FINETUNE,
        domain=LLMDomain.INSTRUCTION,
        split="train",
        conversation_fields=("messages", "conversations"),
        description="System prompt following for chat assistants.",
    )


@register_dataset("longalign")
def longalign() -> LLMDatasetSpec:
    """LongAlign: 3.7k long-context (8k-16k token) English samples."""
    return LLMDatasetSpec(
        name="longalign",
        hf_repo="THUDM/LongAlign-10k",
        stage=TrainingStage.FINETUNE,
        domain=LLMDomain.INSTRUCTION,
        split="train",
        conversation_fields=("messages", "conversations"),
        description="Long-context alignment data for 8k context retention.",
    )


@register_dataset("everyday_conversations")
def everyday_conversations() -> LLMDatasetSpec:
    """Everyday-Conversations: 2.2k casual multi-turn interactions."""
    return LLMDatasetSpec(
        name="everyday_conversations",
        hf_repo="HuggingFaceTB/everyday-conversations-llama3.1-2k",
        stage=TrainingStage.FINETUNE,
        domain=LLMDomain.INSTRUCTION,
        split="train",
        conversation_fields=("messages", "conversations"),
        description="Casual everyday dialogue for natural conversation style.",
    )


@register_dataset("explore_instruct")
def explore_instruct() -> LLMDatasetSpec:
    """Explore-Instruct: 32k rewriting instruction samples."""
    return LLMDatasetSpec(
        name="explore_instruct",
        hf_repo="sqwu/Explore-Instruct-Rewriting",
        stage=TrainingStage.FINETUNE,
        domain=LLMDomain.INSTRUCTION,
        split="train",
        prompt_fields=("instruction", "prompt"),
        response_fields=("response", "output"),
        description="Active exploration rewriting instructions.",
    )


@register_dataset("openhermes25")
def openhermes25() -> LLMDatasetSpec:
    """OpenHermes 2.5: 100k general instruction samples in SmolTalk."""
    return LLMDatasetSpec(
        name="openhermes25",
        hf_repo="teknium/OpenHermes-2.5",
        stage=TrainingStage.FINETUNE,
        domain=LLMDomain.INSTRUCTION,
        split="train",
        conversation_fields=("messages", "conversations"),
        description="Strong general knowledge instruction data.",
    )
