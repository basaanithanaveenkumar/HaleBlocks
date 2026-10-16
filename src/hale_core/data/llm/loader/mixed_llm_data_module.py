"""``MixedLLMDataModule``: Trainer-compatible wrapper for LLM streaming."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

from hale_core.config.sections.webdataset import WebDatasetConfig
from hale_core.data.kinds import TrainingStage
from hale_core.data.llm.loader._build_llm_dataloader import build_llm_dataloader
from hale_core.data.llm.loader._helpers import _collate_llm_batch
from hale_core.data.llm.types import (
    LLMSample,
)


class MixedLLMDataModule:
    """Trainer-compatible wrapper for LLM streaming.

    With ``webdataset.enabled`` the mixture is written once to WebDataset shards (the HF streams
    are only opened when the shards are missing or stale) and training streams the shards. The
    mixture must then be finite: set ``samples_per_dataset`` (or ``webdataset.max_samples``).
    """

    def __init__(
        self,
        *,
        dataset_names: Sequence[str] | None = None,
        stage: TrainingStage | str | None = None,
        cache_dir: str | Path = "~/.cache/hale-blocks/llm",
        samples_per_dataset: int | None = None,
        batch_size: int = 1,
        tokenizer=None,
        webdataset: WebDatasetConfig | None = None,
    ) -> None:
        from hale_core.data.webdataset import WebDatasetAdapter  # lazy: avoids an import cycle
        from hale_core.data.webdataset.fingerprint import fingerprint

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        names = tuple(dataset_names) if dataset_names is not None else None
        fp = fingerprint(
            kind="llm", names=names, stage=str(stage), samples_per_dataset=samples_per_dataset
        )
        self._loader = WebDatasetAdapter(webdataset or WebDatasetConfig()).prepare_iterable(
            lambda: build_llm_dataloader(
                names, stage=stage, cache_dir=cache_dir, samples_per_dataset=samples_per_dataset
            ),
            f"llm-mixture-{fp[:12]}",
            fingerprint=fp,
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
