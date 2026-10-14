"""Write any PyTorch dataset to WebDataset tar shards (once, on rank 0)."""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from loguru import logger
from torch.utils.data import DataLoader, Dataset, IterableDataset
from tqdm import tqdm

from hale_core.config.sections.webdataset import WebDatasetConfig
from hale_core.data.webdataset.dependency import require_webdataset
from hale_core.data.webdataset.manifest import ShardManifest
from hale_core.data.webdataset.sample_codec import SampleCodec
from hale_core.distributed.ray.worker_context import WorkerContext

SHARD_PATTERN = "shard-%06d.tar"

#: A dataset, or a zero-argument callable that builds it (only called if conversion is needed).
DatasetSource = Dataset | Callable[[], Dataset]


def _keep_sample(sample: Any) -> Any:
    """Identity collate (module-level so DataLoader workers can pickle it)."""
    return sample


class WebDatasetConverter:
    """Dataset -> ``<root>/<name>/shard-*.tar`` + ``manifest.json``.

    * Conversion happens **once**: if a valid manifest with the same fingerprint exists, the
      shards are reused (``overwrite: true`` forces a rewrite).
    * In a distributed run only rank 0 writes; the other ranks wait at a barrier and then read
      the same manifest.
    * Shards are written to a temporary directory and renamed into place at the end, so a crash
      never leaves a half-written dataset that looks valid.
    * ``convert_workers > 0`` reads a map-style dataset with that many DataLoader workers (useful
      when ``__getitem__`` is expensive, e.g. rendering or decoding video).

    Args:
        config: the ``data.webdataset`` section.
        codec: sample serialiser (default :class:`SampleCodec`).
    """

    def __init__(self, config: WebDatasetConfig, codec: SampleCodec | None = None) -> None:
        self.config = config
        self.codec = codec or SampleCodec()

    # ------------------------------------------------------------------ public API
    def directory(self, name: str) -> Path:
        """Where the shards of dataset ``name`` live."""
        return Path(self.config.root).expanduser() / name

    def fingerprint(self, dataset: Dataset, explicit: str | None = None) -> str:
        """Identity of the source used to decide whether existing shards can be reused.

        The default (class, length, conversion limits) cannot see changes *inside* a dataset;
        pass ``explicit`` (e.g. a hash of the data config) when the content can change while
        the class and length stay the same.
        """
        if explicit is not None:
            return explicit
        cls = type(dataset)
        length = len(dataset) if hasattr(dataset, "__len__") else "stream"
        return (
            f"{cls.__module__}.{cls.__qualname__}|len={length}|max={self.config.max_samples}"
            f"|per_shard={self.config.samples_per_shard}"
        )

    def convert(
        self, source: DatasetSource, name: str, fingerprint: str | None = None
    ) -> ShardManifest:
        """Make sure valid shards exist for ``source`` and return their manifest.

        ``source`` may be a zero-argument factory instead of a dataset: it is then called only
        when the shards are missing or stale, so expensive datasets (downloads, tokenisation,
        rendering) are not even built on runs that reuse the shards. A factory needs an explicit
        ``fingerprint``.
        """
        context = WorkerContext.current()
        directory = self.directory(name)
        is_factory = callable(source) and not isinstance(source, Dataset)
        if is_factory and fingerprint is None:
            raise ValueError("a dataset factory needs an explicit fingerprint")
        fp = fingerprint if is_factory else self.fingerprint(source, fingerprint)
        if context.is_main:
            self._convert_if_needed(source, name, directory, fp)
        self._barrier()
        manifest = ShardManifest.load(directory)
        if manifest is None or not manifest.is_valid(directory, fp, self.codec.version):
            raise RuntimeError(f"no valid WebDataset shards in {directory} after conversion")
        return manifest

    # ------------------------------------------------------------------ helpers
    def _convert_if_needed(
        self, source: DatasetSource, name: str, directory: Path, fp: str
    ) -> None:
        existing = ShardManifest.load(directory)
        if (
            existing is not None
            and not self.config.overwrite
            and existing.is_valid(directory, fp, self.codec.version)
        ):
            logger.info(
                "webdataset: reusing {} samples in {} shards at {}",
                existing.num_samples,
                len(existing.shards),
                directory,
            )
            return
        wds = require_webdataset()
        dataset = source() if callable(source) and not isinstance(source, Dataset) else source
        tmp = directory.with_name(f".{directory.name}.tmp-{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        written = 0
        with wds.ShardWriter(
            str(tmp / SHARD_PATTERN),
            maxcount=self.config.samples_per_shard,
            maxsize=self.config.max_shard_bytes,
            verbose=0,
        ) as sink:
            for index, sample in enumerate(
                tqdm(self._samples(dataset), desc=f"webdataset:{name}", total=self._total(dataset))
            ):
                sink.write(self.codec.encode(sample, key=f"{index:09d}"))
                written += 1
        if written == 0:
            shutil.rmtree(tmp, ignore_errors=True)
            raise ValueError(f"dataset {name!r} produced no samples; nothing to convert")
        shards = sorted(p.name for p in tmp.glob("shard-*.tar"))
        ShardManifest(
            name=name,
            num_samples=written,
            shards=shards,
            codec_version=self.codec.version,
            fingerprint=fp,
            samples_per_shard=self.config.samples_per_shard,
        ).save(tmp)
        # Atomic-ish swap: readers either see the old complete directory or the new one.
        shutil.rmtree(directory, ignore_errors=True)
        directory.parent.mkdir(parents=True, exist_ok=True)
        tmp.rename(directory)
        logger.info(
            "webdataset: wrote {} samples in {} shards to {}", written, len(shards), directory
        )

    def _total(self, dataset: Dataset) -> int | None:
        if not hasattr(dataset, "__len__"):
            return self.config.max_samples
        n = len(dataset)
        return n if self.config.max_samples is None else min(n, self.config.max_samples)

    def _samples(self, dataset: Dataset) -> Iterator[Any]:
        """Samples in order, read with ``convert_workers`` DataLoader workers when possible."""
        workers = self.config.convert_workers
        if isinstance(dataset, IterableDataset) and workers > 0:
            # An IterableDataset is replicated in every worker unless it shards itself, which
            # would duplicate samples - read it in the main process instead.
            logger.warning("webdataset: IterableDataset is converted with convert_workers=0")
            workers = 0
        loader = DataLoader(
            dataset,
            batch_size=None,  # one sample at a time, no batching
            shuffle=False,
            num_workers=workers,
            collate_fn=_keep_sample,  # keep samples exactly as the dataset returns them
        )
        for index, sample in enumerate(loader):
            if self.config.max_samples is not None and index >= self.config.max_samples:
                break
            yield sample

    @staticmethod
    def _barrier() -> None:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()
