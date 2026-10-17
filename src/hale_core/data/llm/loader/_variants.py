"""Variants and registrations of :mod:`hale_core.data.llm.loader` (Sequential LLM dataloader with
streaming and prefetch).
"""

from __future__ import annotations

from pathlib import Path

from hale_core.data.llm.loader.sequential_llm_data_loader import SequentialLLMDataLoader
from hale_core.data.llm.types import (
    SMOLLM2_PRETRAIN_STAGES,
)


def build_smollm2_stage_loader(
    stage: int,
    *,
    cache_dir: str | Path = "~/.cache/hale-blocks/llm",
    samples_per_dataset: int | None = None,
) -> SequentialLLMDataLoader:
    """Loader for SmolLM2 training stage ``stage`` (its published dataset mixture).

    Args:
        stage: SmolLM2 stage number.
        cache_dir: local cache root.
        samples_per_dataset: cap per dataset (``None`` = stream everything).
    """
    mixture = SMOLLM2_PRETRAIN_STAGES[stage]
    names = [name for name, _weight in mixture.datasets]
    return SequentialLLMDataLoader(
        dataset_names=names,
        cache_dir=Path(cache_dir),
        samples_per_dataset=samples_per_dataset,
    )
