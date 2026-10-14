"""``manifest.json``: what a directory of shards contains and where it came from."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

MANIFEST_NAME = "manifest.json"


@dataclass
class ShardManifest:
    """Description of one converted dataset.

    Attributes:
        name: dataset name (= sub-directory of ``WebDatasetConfig.root``).
        num_samples: samples written over all shards.
        shards: shard file names, relative to the manifest's directory, in write order.
        codec_version: :class:`SampleCodec` version used to write the samples.
        fingerprint: identifies the source dataset; a different fingerprint invalidates the shards.
        samples_per_shard: the limit used while writing.
    """

    name: str
    num_samples: int
    shards: list[str]
    codec_version: int
    fingerprint: str
    samples_per_shard: int
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ io
    def save(self, directory: str | Path) -> Path:
        path = Path(directory) / MANIFEST_NAME
        path.write_text(json.dumps(asdict(self), indent=2))
        return path

    @classmethod
    def load(cls, directory: str | Path) -> ShardManifest | None:
        """The manifest in ``directory``, or ``None`` if there is none (or it is unreadable)."""
        path = Path(directory) / MANIFEST_NAME
        if not path.is_file():
            return None
        try:
            return cls(**json.loads(path.read_text()))
        except (json.JSONDecodeError, TypeError):
            return None

    # ------------------------------------------------------------------ queries
    def urls(self, directory: str | Path) -> list[str]:
        """Absolute paths of the shards (what ``webdataset.WebDataset`` reads)."""
        return [str(Path(directory) / shard) for shard in self.shards]

    def is_valid(self, directory: str | Path, fingerprint: str, codec_version: int) -> bool:
        """Shards exist and were written from the same source with the same codec."""
        return (
            self.fingerprint == fingerprint
            and self.codec_version == codec_version
            and self.num_samples > 0
            and all((Path(directory) / shard).is_file() for shard in self.shards)
        )
