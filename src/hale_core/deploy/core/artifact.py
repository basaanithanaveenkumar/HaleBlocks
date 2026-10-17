"""A file or directory produced by a stage, with provenance."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Artifact:
    """Attributes:
    kind: what downstream stages ask for (``"onnx"``, ``"tensorrt_engine"``,
        ``"mlx_function"`` ...). A later artifact of the same kind supersedes an earlier one
        (e.g. a simplified or quantized ONNX replaces the raw export).
    path: file or directory on disk.
    producer: name of the stage that produced it.
    metadata: free-form facts (precision, opset, sizes, device ...).
    """

    kind: str
    path: Path
    producer: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        """Size of the artifact on disk (files summed for a directory)."""
        if self.path.is_dir():
            return sum(p.stat().st_size for p in self.path.rglob("*") if p.is_file())
        return self.path.stat().st_size if self.path.exists() else 0

    def sha256(self, chunk: int = 1 << 20) -> str | None:
        """Streaming checksum of a file artifact (``None`` for directories)."""
        if not self.path.is_file():
            return None
        digest = hashlib.sha256()
        with self.path.open("rb") as fh:
            while block := fh.read(chunk):
                digest.update(block)
        return digest.hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable description (kind, path, producer, metadata)."""
        data = asdict(self)
        data["path"] = str(self.path)
        data["size_bytes"] = self.size_bytes
        data["sha256"] = self.sha256()
        return data
