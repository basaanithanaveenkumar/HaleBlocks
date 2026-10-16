"""``LLMDatasetSpec``: LLMDatasetSpec (LLM dataset sample skeleton (SmolLM2 paper))."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hale_core.data.kinds import DatasetKind, LLMDomain, TrainingStage


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
