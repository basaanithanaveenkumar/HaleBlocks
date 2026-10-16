"""``SequentialVLMDataLoader``: Iterate datasets one after another with bounded prefetch + cache."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from hale_core.data.kinds import DatasetKind
from hale_core.data.vlm.cache import MediaCache
from hale_core.data.vlm.parse import default_row_parser
from hale_core.data.vlm.types import VLMDatasetSpec, VLMSample
from hale_core.registry import get_dataset


@dataclass
class SequentialVLMDataLoader:
    """Iterate datasets one after another with bounded prefetch + cache."""

    dataset_names: Sequence[str]
    cache_dir: Path = Path("~/.cache/hale-blocks/vlm")
    cache_max_bytes: int = 2 * 1024**3
    samples_per_dataset: int | None = None
    prefetch_workers: int = 4
    prefetch_queue: int = 8

    def __post_init__(self) -> None:
        self.cache_dir = self.cache_dir.expanduser().resolve()
        self._cache = MediaCache(self.cache_dir / "media", max_bytes=self.cache_max_bytes)

    def __iter__(self) -> Iterator[VLMSample]:
        for name in self.dataset_names:
            spec = get_dataset(name)
            if spec.kind != DatasetKind.VLM:
                raise TypeError(f"{name!r} is not a VLM dataset (kind={spec.kind})")
            yield from self._iter_dataset(spec)

    def _iter_dataset(self, spec: VLMDatasetSpec) -> Iterator[VLMSample]:
        logger.info("sequential VLM loader starting dataset={}", spec.name)
        stream = spec.open_stream(cache_dir=self.cache_dir)
        pending: deque[Future[VLMSample]] = deque()

        with ThreadPoolExecutor(max_workers=self.prefetch_workers) as pool:
            for idx, row in enumerate(stream):
                if self.samples_per_dataset is not None and idx >= self.samples_per_dataset:
                    break

                pending.append(
                    pool.submit(
                        default_row_parser,
                        row,
                        spec,
                        cache=self._cache,
                        sample_idx=idx,
                    )
                )
                if len(pending) >= self.prefetch_queue:
                    yield pending.popleft().result()

            while pending:
                yield pending.popleft().result()

    def __len__(self) -> int:
        if self.samples_per_dataset is None:
            raise TypeError("length requires samples_per_dataset")
        return len(self.dataset_names) * self.samples_per_dataset
