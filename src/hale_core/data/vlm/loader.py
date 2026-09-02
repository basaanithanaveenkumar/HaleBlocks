"""Sequential VLM dataloader with on-demand media caching."""

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
from hale_core.data.vlm.types import DEFAULT_VLM_MIXTURE, VLMDatasetSpec, VLMSample
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


def build_vlm_dataloader(
    dataset_names: Sequence[str] | None = None,
    *,
    cache_dir: str | Path = "~/.cache/hale-blocks/vlm",
    cache_max_bytes: int = 2 * 1024**3,
    samples_per_dataset: int | None = None,
    prefetch_workers: int = 4,
    prefetch_queue: int = 8,
) -> SequentialVLMDataLoader:
    names = tuple(dataset_names or DEFAULT_VLM_MIXTURE)
    return SequentialVLMDataLoader(
        dataset_names=names,
        cache_dir=Path(cache_dir),
        cache_max_bytes=cache_max_bytes,
        samples_per_dataset=samples_per_dataset,
        prefetch_workers=prefetch_workers,
        prefetch_queue=prefetch_queue,
    )


class MixedVLMDataModule:
    """Trainer-compatible wrapper around :class:`SequentialVLMDataLoader`."""

    def __init__(
        self,
        *,
        dataset_names: Sequence[str] | None = None,
        cache_dir: str | Path = "~/.cache/hale-blocks/vlm",
        cache_max_bytes: int = 2 * 1024**3,
        samples_per_dataset: int | None = None,
        prefetch_workers: int = 4,
        batch_size: int = 1,
        tokenizer=None,
    ) -> None:
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self._loader = build_vlm_dataloader(
            dataset_names,
            cache_dir=cache_dir,
            cache_max_bytes=cache_max_bytes,
            samples_per_dataset=samples_per_dataset,
            prefetch_workers=prefetch_workers,
        )

    def train_loader(self) -> Iterator[dict]:
        batch: list[VLMSample] = []
        for sample in self._loader:
            batch.append(sample)
            if len(batch) >= self.batch_size:
                yield _collate_vlm_batch(batch)
                batch = []
        if batch:
            yield _collate_vlm_batch(batch)

    def val_loader(self, train_loader):
        return train_loader

    def viz_prompt(self, step: int) -> str:
        return f"vlm-mixture step={step}"


def _collate_vlm_batch(samples: list[VLMSample]) -> dict:
    return {
        "samples": samples,
        "text": [s.text for s in samples],
        "modality": [s.modality.value for s in samples],
        "dataset": [s.dataset for s in samples],
    }
