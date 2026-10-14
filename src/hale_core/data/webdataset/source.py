"""Stream converted shards back as a PyTorch ``IterableDataset``."""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from torch.utils.data import IterableDataset

from hale_core.config.sections.webdataset import WebDatasetConfig
from hale_core.data.webdataset.dependency import require_webdataset
from hale_core.data.webdataset.manifest import ShardManifest
from hale_core.data.webdataset.sample_codec import SampleCodec
from hale_core.data.webdataset.sized_stream import SizedStream
from hale_core.distributed.ray.worker_context import WorkerContext


class WebDatasetSource:
    """Build the ``webdataset`` pipeline for one converted dataset.

    Training pipeline (``training=True``)::

        shards ─► [resample shards | shuffle + split by rank & worker] ─► tar members
               ─► shuffle buffer ─► SampleCodec.decode ─► fixed-length epoch

    * **Single process**: every shard is read exactly once per epoch, in shuffled order.
    * **Several ranks** (``resample: auto``): each rank samples shards with replacement and an
      epoch is ``num_samples / world_size`` samples on *every* rank. Equal step counts are what
      DDP needs - splitting shards across ranks would give ranks different lengths and hang the
      gradient all-reduce at the end of the shortest one.
    * **Evaluation** (``training=False``): no shuffling, no resampling; shards are split across
      ranks and workers so every sample is seen once in total.

    The dataset has ``__len__`` (samples per rank), so ``len(DataLoader)`` keeps working.

    Args:
        manifest: description of the shards.
        directory: directory holding the shards and manifest.
        config: the ``data.webdataset`` section.
        codec: must match the codec used to write the shards.
    """

    def __init__(
        self,
        manifest: ShardManifest,
        directory: str | Path,
        config: WebDatasetConfig,
        codec: SampleCodec | None = None,
    ) -> None:
        self.manifest = manifest
        self.directory = Path(directory)
        self.config = config
        self.codec = codec or SampleCodec()

    def build(self, *, training: bool = True, loader_workers: int = 0) -> IterableDataset:
        """The streaming dataset.

        Args:
            training: shuffle / resample for training (``False`` for evaluation).
            loader_workers: ``num_workers`` of the DataLoader that will iterate it; used to
                split the per-rank epoch length between workers when resampling.
        """
        wds = require_webdataset()
        ctx = WorkerContext.current()
        urls = self.manifest.urls(self.directory)
        per_rank = max(1, self.manifest.num_samples // ctx.world_size)
        shuffle = training and self.config.shuffle
        resample = training and (
            self.config.resample == "always"
            or (self.config.resample == "auto" and ctx.world_size > 1)
        )

        if resample:
            # Each rank/worker draws shards with replacement from all shards (seeded per rank).
            # Nothing to split: every rank and worker draws its own shards, so both splitters are
            # off (webdataset's default node splitter refuses to run with WORLD_SIZE > 1).
            dataset = wds.WebDataset(
                urls,
                resampled=True,
                shardshuffle=False,
                nodesplitter=None,
                workersplitter=None,
                empty_check=False,
                seed=self.config.seed + ctx.rank,
            )
        else:
            self._warn_if_too_few_shards(len(urls), ctx.world_size, loader_workers)
            dataset = wds.WebDataset(
                urls,
                shardshuffle=len(urls) if shuffle else False,
                nodesplitter=wds.split_by_node,
                workersplitter=wds.split_by_worker,
                empty_check=False,
                seed=self.config.seed,
            )
        if shuffle and self.config.shuffle_buffer > 0:
            dataset = dataset.shuffle(self.config.shuffle_buffer)
        dataset = dataset.map(self.codec.decode)
        if resample:
            # Resampled streams are infinite: cut an epoch. with_epoch counts per DataLoader worker.
            dataset = dataset.with_epoch(max(1, per_rank // max(1, loader_workers)))
        return SizedStream(dataset, per_rank)

    def _warn_if_too_few_shards(self, num_shards: int, world_size: int, workers: int) -> None:
        readers = world_size * max(1, workers)
        if num_shards < readers:
            logger.warning(
                "webdataset: {} shard(s) for {} reader(s) (ranks x workers): some readers get no "
                "data. Lower samples_per_shard to write more shards.",
                num_shards,
                readers,
            )
