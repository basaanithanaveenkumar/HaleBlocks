"""``VLMSample``: VLMSample (VLM dataset sample skeleton)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hale_core.data.kinds import VLMModality


@dataclass
class VLMSample:
    dataset: str
    modality: VLMModality
    text: str | None = None
    images: list[Path] = field(default_factory=list)
    video: Path | None = None
    conversations: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_vision(self) -> bool:
        return bool(self.images or self.video)
