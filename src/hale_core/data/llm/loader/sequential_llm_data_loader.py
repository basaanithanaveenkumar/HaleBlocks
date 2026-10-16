"""``SequentialLLMDataLoader``: Stream datasets sequentially with on-demand HF download."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from hale_core.data.kinds import DatasetKind
from hale_core.data.llm.parse import default_row_parser
from hale_core.data.llm.types import (
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
