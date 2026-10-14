"""The single entry point used by data pipelines: dataset in, dataset (maybe WebDataset) out."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from torch.utils.data import Dataset

from hale_core.config.sections.webdataset import WebDatasetConfig
from hale_core.data.webdataset.converter import DatasetSource, WebDatasetConverter
from hale_core.data.webdataset.sample_codec import SampleCodec
from hale_core.data.webdataset.sized_stream import SizedStream
from hale_core.data.webdataset.source import WebDatasetSource


class WebDatasetAdapter:
    """Flag-driven switch between a dataset and its WebDataset version.

    ``prepare(dataset, name)`` returns ``dataset`` unchanged when ``config.enabled`` is false.
    When it is true it converts the dataset to shards (first run only) and returns the streamed
    :class:`torch.utils.data.IterableDataset`, whose samples are identical to the original ones
    (same structure, dtypes and values). Use one adapter per run and one ``name`` per split::

        adapter = WebDatasetAdapter(cfg.data.webdataset)
        train_set = adapter.prepare(train_set, "train")
        val_set = adapter.prepare(val_set, "val", training=False)

    Args:
        config: the ``data.webdataset`` section.
        codec: sample serialiser shared by the converter and the reader.
    """

    def __init__(self, config: WebDatasetConfig, codec: SampleCodec | None = None) -> None:
        self.config = config
        self.codec = codec or SampleCodec()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def prepare(
        self,
        dataset: DatasetSource,
        name: str,
        *,
        training: bool = True,
        loader_workers: int = 0,
        fingerprint: str | None = None,
    ) -> Dataset:
        """``dataset`` itself, or its WebDataset stream when the flag is on.

        Args:
            dataset: any map-style or iterable dataset whose samples :class:`SampleCodec` can
                store - or a zero-argument factory building it. With the flag off a factory is
                called and its dataset returned; with the flag on it is only called when the
                shards must be (re)written, so give it an explicit ``fingerprint``.
            name: shard sub-directory, e.g. ``"train"``; must be unique per split / source.
            training: shuffle and (multi-rank) resample; ``False`` for evaluation.
            loader_workers: ``num_workers`` of the DataLoader that will read it.
            fingerprint: explicit source identity (see :meth:`WebDatasetConverter.fingerprint`).
        """
        if not self.config.enabled:
            return dataset() if callable(dataset) and not isinstance(dataset, Dataset) else dataset
        converter = WebDatasetConverter(self.config, self.codec)
        manifest = converter.convert(dataset, name, fingerprint)
        source = WebDatasetSource(manifest, converter.directory(name), self.config, self.codec)
        return source.build(training=training, loader_workers=loader_workers)

    def prepare_iterable(
        self,
        factory: Callable[[], Iterable[Any]],
        name: str,
        *,
        fingerprint: str,
        length: int | None = None,
        training: bool = True,
        loader_workers: int = 0,
    ) -> Iterable[Any]:
        """:meth:`prepare` for sources that are plain iterables rather than torch datasets.

        Streaming mixtures (e.g. :class:`~hale_core.data.vlm.loader.SequentialVLMDataLoader`)
        are iterables of samples. With the flag off this returns ``factory()`` unchanged. With
        it on, ``factory`` is only called when the shards must be (re)written - a run that
        reuses shards never opens the original streams - and the WebDataset stream is returned.

        Args:
            factory: zero-argument callable building the iterable.
            name: shard sub-directory (unique per source / split).
            fingerprint: identity of the source's content (required: an iterable has no class
                or length the converter could use to detect changes).
            length: samples to convert; defaults to ``len(iterable)``. An unbounded stream needs
                either this or ``WebDatasetConfig.max_samples``.
            training: shuffle and (multi-rank) resample; ``False`` for evaluation.
            loader_workers: ``num_workers`` of the DataLoader that will read the stream.
        """
        if not self.config.enabled:
            return factory()

        def as_dataset() -> SizedStream:
            iterable = factory()
            size = length if length is not None else self._length(iterable)
            return SizedStream(iterable, size)

        return self.prepare(
            as_dataset,
            name,
            training=training,
            loader_workers=loader_workers,
            fingerprint=fingerprint,
        )

    def _length(self, iterable: Iterable[Any]) -> int:
        try:
            return len(iterable)  # type: ignore[arg-type]
        except TypeError:
            if self.config.max_samples is None:
                raise ValueError(
                    "cannot convert an unbounded stream to WebDataset: pass length=..., give the "
                    "loader a size (e.g. samples_per_dataset) or set data.webdataset.max_samples"
                ) from None
            return self.config.max_samples


def maybe_webdataset(
    dataset: Dataset,
    config: WebDatasetConfig,
    name: str,
    **kwargs,
) -> Dataset:
    """Functional shortcut for ``WebDatasetAdapter(config).prepare(dataset, name, **kwargs)``."""
    return WebDatasetAdapter(config).prepare(dataset, name, **kwargs)
