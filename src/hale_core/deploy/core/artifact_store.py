"""Kind-indexed store of the artifacts a pipeline run has produced so far."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from loguru import logger

from hale_core.deploy.core.artifact import Artifact
from hale_core.deploy.exceptions import PipelineError


class ArtifactStore:
    """``kind → history`` map; ``get`` returns the newest artifact of a kind in O(1).

    Keeping the history (not just the latest) lets the manifest show the full lineage, e.g.
    ``onnx: model.onnx → model.sim.onnx → model.fp16.onnx``.
    """

    def __init__(self) -> None:
        self._history: dict[str, list[Artifact]] = defaultdict(list)

    def put(self, artifact: Artifact) -> Artifact:
        """Record ``artifact`` as the newest of its kind and return it."""
        self._history[artifact.kind].append(artifact)
        logger.debug("artifact {} <- {} ({})", artifact.kind, artifact.path, artifact.producer)
        return artifact

    def get(self, kind: str) -> Artifact | None:
        """Newest artifact of ``kind``, or ``None``."""
        history = self._history.get(kind)
        return history[-1] if history else None

    def require(self, kind: str) -> Artifact:
        """Newest artifact of ``kind``.

        Raises:
            PipelineError: if no stage produced that kind.
        """
        artifact = self.get(kind)
        if artifact is None:
            raise PipelineError(
                f"no {kind!r} artifact has been produced; available: {sorted(self.kinds())}"
            )
        return artifact

    def kinds(self) -> set[str]:
        """Every artifact kind produced so far."""
        return {k for k, v in self._history.items() if v}

    def history(self, kind: str) -> list[Artifact]:
        """All artifacts of ``kind``, oldest first."""
        return list(self._history.get(kind, []))

    def all(self) -> list[Artifact]:
        """Every artifact in production order."""
        return [a for items in self._history.values() for a in items]

    def write_manifest(self, path: Path) -> Path:
        """Write all artifacts as JSON to ``path`` and return it."""
        manifest = {kind: [a.to_dict() for a in items] for kind, items in self._history.items()}
        path.write_text(json.dumps(manifest, indent=2, default=str))
        return path
