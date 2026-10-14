"""WebDataset sub-config: convert any dataset to tar shards and stream it back."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from hale_core.config.sections.common import StrictModel


class WebDatasetConfig(StrictModel):
    """Turn any PyTorch dataset into WebDataset shards with a single flag.

    With ``enabled: true`` the dataset that the data pipeline would normally use is written
    once to POSIX tar shards (``<root>/<name>/shard-000000.tar`` ...) together with a
    ``manifest.json``; training then streams those shards with ``webdataset`` (sequential
    reads, shuffled shards + an in-memory shuffle buffer, automatic splitting across ranks and
    DataLoader workers). Later runs reuse the shards when the manifest matches. Disabled by
    default, so existing pipelines are untouched.

    YAML example::

        data:
          webdataset:
            enabled: true
            root: /fast_disk/shards
            samples_per_shard: 2000
            convert_workers: 8
            shuffle_buffer: 5000
    """

    enabled: bool = False
    """Convert the dataset to WebDataset shards and train from them. Requires the
    ``webdataset`` extra (``pip install hale-blocks[webdataset]``)."""

    # ── Conversion (writing shards) ──────────────────────────────────────────
    root: str = "data/webdataset"
    """Directory holding one sub-directory of shards per dataset name (local, NFS ...)."""
    samples_per_shard: int = Field(default=1000, ge=1)
    """Maximum samples per tar shard."""
    max_shard_bytes: int = Field(default=1_000_000_000, ge=1)
    """Maximum size of one tar shard in bytes (whichever limit is hit first closes the shard)."""
    convert_workers: int = Field(default=0, ge=0)
    """DataLoader workers used to read the source dataset while converting (0 = main process)."""
    max_samples: int | None = Field(default=None, ge=1)
    """Convert only the first ``max_samples`` samples (useful for smoke tests)."""
    overwrite: bool = False
    """Re-convert even if valid shards for the same source already exist."""

    # ── Reading (streaming shards back) ──────────────────────────────────────
    shuffle: bool = True
    """Shuffle the shard order and samples within a buffer (training splits)."""
    shuffle_buffer: int = Field(default=1000, ge=0)
    """Size of the in-memory sample shuffle buffer (0 disables sample shuffling)."""
    resample: Literal["auto", "always", "never"] = "auto"
    """Sample shards with replacement. ``auto`` = only when several ranks train, which gives
    every rank the same number of steps per epoch (required to avoid DDP hangs) even if the
    shard count is not divisible by the number of ranks."""
    seed: int = 0
    """Seed of the shard/sample shuffling."""
