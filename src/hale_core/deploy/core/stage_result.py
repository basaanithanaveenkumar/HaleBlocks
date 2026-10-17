"""Outcome of one stage in a pipeline run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from hale_core.deploy.core.artifact import Artifact

Status = Literal["ok", "skipped", "failed"]


@dataclass
class StageResult:
    """Outcome of one stage: status, duration, produced artifacts and message."""

    name: str
    stage_type: str
    status: Status
    seconds: float = 0.0
    artifacts: list[Artifact] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form of the result."""
        return {
            "name": self.name,
            "type": self.stage_type,
            "status": self.status,
            "seconds": round(self.seconds, 3),
            "artifacts": [str(a.path) for a in self.artifacts],
            "message": self.message,
        }
