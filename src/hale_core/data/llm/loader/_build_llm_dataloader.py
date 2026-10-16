"""``build_llm_dataloader``: build_llm_dataloader (Sequential LLM dataloader with streaming and prefetch)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from hale_core.data.kinds import TrainingStage
from hale_core.data.llm.loader._helpers import _all_pretrain_dataset_names
from hale_core.data.llm.loader.sequential_llm_data_loader import SequentialLLMDataLoader
from hale_core.data.llm.types import (
    DEFAULT_SMOLLM2_RL_MIXTURE,
    DEFAULT_SMOLLM2_SFT_MIXTURE,
)


def build_llm_dataloader(
    dataset_names: Sequence[str] | None = None,
    *,
    stage: TrainingStage | str | None = None,
    cache_dir: str | Path = "~/.cache/hale-blocks/llm",
    samples_per_dataset: int | None = None,
    prefetch_workers: int = 4,
    prefetch_queue: int = 16,
) -> SequentialLLMDataLoader:
    if dataset_names is None:
        if stage in (TrainingStage.PRETRAIN, "pretrain"):
            dataset_names = _all_pretrain_dataset_names()
        elif stage in (TrainingStage.FINETUNE, "finetune", "sft"):
            dataset_names = DEFAULT_SMOLLM2_SFT_MIXTURE
        elif stage in (TrainingStage.RL, "rl", "dpo"):
            dataset_names = DEFAULT_SMOLLM2_RL_MIXTURE
        else:
            dataset_names = _all_pretrain_dataset_names()
    return SequentialLLMDataLoader(
        dataset_names=tuple(dataset_names),
        cache_dir=Path(cache_dir),
        samples_per_dataset=samples_per_dataset,
        prefetch_workers=prefetch_workers,
        prefetch_queue=prefetch_queue,
    )
