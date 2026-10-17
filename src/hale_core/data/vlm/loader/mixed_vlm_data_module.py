"""``MixedVLMDataModule``: Trainer-compatible wrapper around :class:`SequentialVLMDataLoader`."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

from hale_core.config.sections.webdataset import WebDatasetConfig
from hale_core.data.vlm.loader._build_vlm_dataloader import build_vlm_dataloader
from hale_core.data.vlm.loader._helpers import _collate_vlm_batch
from hale_core.data.vlm.types import DEFAULT_VLM_MIXTURE, VLMSample


class MixedVLMDataModule:
    """Trainer-compatible wrapper around :class:`SequentialVLMDataLoader`.

    With ``webdataset.enabled`` the mixture is written once to WebDataset shards (the HF streams
    are only opened when the shards are missing or stale) and training streams the shards. The
    mixture must then be finite: set ``samples_per_dataset`` (or ``webdataset.max_samples``).
    Image / video fields are :class:`~pathlib.Path` objects into ``cache_dir``; they are stored
    as paths, so keep that media cache next to the shards.
    """

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
        webdataset: WebDatasetConfig | None = None,
    ) -> None:
        from hale_core.data.webdataset import WebDatasetAdapter  # lazy: avoids an import cycle
        from hale_core.data.webdataset.fingerprint import fingerprint

        self.tokenizer = tokenizer
        self.batch_size = batch_size
        names = tuple(dataset_names or DEFAULT_VLM_MIXTURE)
        fp = fingerprint(kind="vlm", names=names, samples_per_dataset=samples_per_dataset)
        self._loader = WebDatasetAdapter(webdataset or WebDatasetConfig()).prepare_iterable(
            lambda: build_vlm_dataloader(
                names,
                cache_dir=cache_dir,
                cache_max_bytes=cache_max_bytes,
                samples_per_dataset=samples_per_dataset,
                prefetch_workers=prefetch_workers,
            ),
            f"vlm-mixture-{fp[:12]}",
            fingerprint=fp,
        )

    def train_loader(self) -> Iterator[dict]:
        """Yield collated training batches of ``batch_size`` samples from the mixture."""
        batch: list[VLMSample] = []
        for sample in self._loader:
            batch.append(sample)
            if len(batch) >= self.batch_size:
                yield _collate_vlm_batch(batch)
                batch = []
        if batch:
            yield _collate_vlm_batch(batch)

    def val_loader(self, train_loader):
        """Validation batches: this module reuses the training stream (no held-out split)."""
        return train_loader

    def viz_prompt(self, step: int) -> str:
        """Label shown in visualisations for ``step``."""
        return f"vlm-mixture step={step}"
