"""LLM dataset sample skeleton (SmolLM2 paper)."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hale_core.data.kinds import DatasetKind, LLMDomain, TrainingStage


@dataclass
class LLMSample:
    """Normalized text sample for pretraining, SFT, or preference learning."""

    dataset: str
    stage: TrainingStage
    domain: LLMDomain
    text: str | None = None
    prompt: str | None = None
    response: str | None = None
    conversations: list[dict[str, Any]] | None = None
    chosen: str | None = None
    rejected: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMDatasetSpec:
    name: str
    hf_repo: str
    stage: TrainingStage
    domain: LLMDomain
    kind: DatasetKind = DatasetKind.LLM
    hf_subset: str | None = None
    split: str = "train"
    streaming: bool = True
    text_fields: tuple[str, ...] = ("text", "content", "body")
    prompt_fields: tuple[str, ...] = ("instruction", "prompt", "question", "input")
    response_fields: tuple[str, ...] = ("response", "answer", "output", "completion")
    conversation_fields: tuple[str, ...] = ("messages", "conversations")
    chosen_fields: tuple[str, ...] = ("chosen", "chosen_response")
    rejected_fields: tuple[str, ...] = ("rejected", "rejected_response")
    source_filter: str | None = None
    description: str = ""
    smollm2_stage: int | None = None
    mix_weight: float | None = None

    def open_stream(self, *, cache_dir: Path) -> Iterator[dict[str, Any]]:
        from hale_core.data.llm.hf import open_hf_stream

        return open_hf_stream(self, cache_dir=cache_dir)


@dataclass(frozen=True)
class StageMixture:
    stage: int
    datasets: tuple[tuple[str, float], ...]
    description: str = ""


SMOLLM2_PRETRAIN_STAGES: dict[int, StageMixture] = {
    1: StageMixture(
        stage=1,
        description="0-6T tokens: 90% English web (60/40 FineWeb-Edu/DCLM), 10% StarCoderData",
        datasets=(
            ("fineweb_edu", 0.54),
            ("dclm", 0.36),
            ("starcoderdata", 0.10),
        ),
    ),
    2: StageMixture(
        stage=2,
        description="6-8T tokens: 75% web, 20% code, 5% OpenWebMath",
        datasets=(
            ("fineweb_edu", 0.45),
            ("dclm", 0.30),
            ("starcoderdata", 0.20),
            ("openwebmath", 0.05),
        ),
    ),
    3: StageMixture(
        stage=3,
        description="8-10T tokens: 74% web (40/60 FW-Edu/DCLM), 16% Stack-Edu, 10% math",
        datasets=(
            ("fineweb_edu", 0.296),
            ("dclm", 0.444),
            ("stack_edu", 0.16),
            ("openwebmath", 0.05),
            ("infimm_webmath", 0.05),
        ),
    ),
    4: StageMixture(
        stage=4,
        description="10-11T decay: 58% web, 24% Stack-Edu, 14% math, 4% Cosmopedia",
        datasets=(
            ("fineweb_edu", 0.232),
            ("dclm", 0.348),
            ("stack_edu", 0.24),
            ("finemath4_plus", 0.10),
            ("infi_webmath3_plus", 0.03),
            ("openwebmath", 0.008),
            ("aug_gsm8k", 0.002),
            ("cosmopedia_v2", 0.04),
        ),
    ),
}

DEFAULT_SMOLLM2_SFT_MIXTURE: tuple[str, ...] = (
    "smoltalk",
    "magpie_ultra",
    "smol_constraint",
    "smol_rewrite",
    "smol_summarization",
    "numinamath_cot",
    "metamathqa",
    "self_oss_starcoder2_instruct",
    "apigen_function_calling",
    "systemchats2",
    "longalign",
    "everyday_conversations",
    "explore_instruct",
    "openhermes25",
)

DEFAULT_SMOLLM2_RL_MIXTURE: tuple[str, ...] = (
    "ultrafeedback",
    "ultrainteract",
    "capybara",
    "orca_dpo",
)
