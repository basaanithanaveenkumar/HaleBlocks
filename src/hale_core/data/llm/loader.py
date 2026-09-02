"""Sequential LLM dataloader with streaming and prefetch."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from hale_core.data.kinds import DatasetKind, TrainingStage
from hale_core.data.llm.parse import default_row_parser
from hale_core.data.llm.types import (
    DEFAULT_SMOLLM2_RL_MIXTURE,
    DEFAULT_SMOLLM2_SFT_MIXTURE,
    SMOLLM2_PRETRAIN_STAGES,
    LLMDatasetSpec,
    LLMSample,
)
from hale_core.registry import get_dataset


@dataclass
class SequentialLLMDataLoader:
    """Stream datasets sequentially with on-demand HF download."""

    dataset_names: Sequence[str]
    cache_dir: Path = Path("~/.cache/hale-blocks/llm")
    samples_per_dataset: int | None = None
    prefetch_workers: int = 4
    prefetch_queue: int = 16

    def __post_init__(self) -> None:
        self.cache_dir = self.cache_dir.expanduser().resolve()

    def __iter__(self) -> Iterator[LLMSample]:
        for name in self.dataset_names:
            spec = get_dataset(name)
            if getattr(spec, "kind", None) != DatasetKind.LLM:
                raise TypeError(f"{name!r} is not an LLM dataset")
            yield from self._iter_dataset(spec)

    def _iter_dataset(self, spec: LLMDatasetSpec) -> Iterator[LLMSample]:
        logger.info("sequential LLM loader starting dataset={} stage={}", spec.name, spec.stage)
        stream = spec.open_stream(cache_dir=self.cache_dir)
        pending: deque[Future[LLMSample]] = deque()

        with ThreadPoolExecutor(max_workers=self.prefetch_workers) as pool:
            for idx, row in enumerate(stream):
                if self.samples_per_dataset is not None and idx >= self.samples_per_dataset:
                    break
                pending.append(pool.submit(default_row_parser, row, spec, sample_idx=idx))
                if len(pending) >= self.prefetch_queue:
                    yield pending.popleft().result()
            while pending:
                yield pending.popleft().result()

    def __len__(self) -> int:
        if self.samples_per_dataset is None:
            raise TypeError("length requires samples_per_dataset")
        return len(self.dataset_names) * self.samples_per_dataset


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


def build_smollm2_stage_loader(
    stage: int,
    *,
    cache_dir: str | Path = "~/.cache/hale-blocks/llm",
    samples_per_dataset: int | None = None,
) -> SequentialLLMDataLoader:
    mixture = SMOLLM2_PRETRAIN_STAGES[stage]
    names = [name for name, _weight in mixture.datasets]
    return SequentialLLMDataLoader(
        dataset_names=names,
        cache_dir=Path(cache_dir),
        samples_per_dataset=samples_per_dataset,
    )


def _all_pretrain_dataset_names() -> tuple[str, ...]:
    names: list[str] = []
    for mixture in SMOLLM2_PRETRAIN_STAGES.values():
        for name, _weight in mixture.datasets:
            if name not in names:
                names.append(name)
    return tuple(names)


class MixedLLMDataModule:
    """Trainer-compatible wrapper for LLM streaming."""

    def __init__(
        self,
        *,
        dataset_names: Sequence[str] | None = None,
        stage: TrainingStage | str | None = None,
        cache_dir: str | Path = "~/.cache/hale-blocks/llm",
        samples_per_dataset: int | None = None,
        batch_size: int = 1,
        tokenizer=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self._loader = build_llm_dataloader(
            dataset_names,
            stage=stage,
            cache_dir=cache_dir,
            samples_per_dataset=samples_per_dataset,
        )

    def train_loader(self) -> Iterator[dict]:
        batch: list[LLMSample] = []
        for sample in self._loader:
            batch.append(sample)
            if len(batch) >= self.batch_size:
                yield _collate_llm_batch(batch)
                batch = []
        if batch:
            yield _collate_llm_batch(batch)

    def val_loader(self, train_loader):
        return train_loader

    def viz_prompt(self, step: int) -> str:
        return f"llm-mixture step={step}"


def _collate_llm_batch(samples: list[LLMSample]) -> dict:
    return {
        "samples": samples,
        "text": [s.text for s in samples],
        "prompt": [s.prompt for s in samples],
        "response": [s.response for s in samples],
        "dataset": [s.dataset for s in samples],
        "stage": [s.stage.value for s in samples],
    }
