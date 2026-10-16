"""Shared helpers of :mod:`hale_core.data.llm.loader` (Sequential LLM dataloader with streaming and prefetch)."""

from __future__ import annotations

from hale_core.data.llm.types import (
    SMOLLM2_PRETRAIN_STAGES,
    LLMSample,
)


def _all_pretrain_dataset_names() -> tuple[str, ...]:
    names: list[str] = []
    for mixture in SMOLLM2_PRETRAIN_STAGES.values():
        for name, _weight in mixture.datasets:
            if name not in names:
                names.append(name)
    return tuple(names)


def _collate_llm_batch(samples: list[LLMSample]) -> dict:
    return {
        "samples": samples,
        "text": [s.text for s in samples],
        "prompt": [s.prompt for s in samples],
        "response": [s.response for s in samples],
        "dataset": [s.dataset for s in samples],
        "stage": [s.stage.value for s in samples],
    }
