"""``LLMSample``: Normalized text sample for pretraining, SFT, or preference learning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hale_core.data.kinds import LLMDomain, TrainingStage


@dataclass
class LLMSample:
    """Normalized text sample for pretraining, SFT, or preference learning."""

    dataset: str
    stage: TrainingStage
    domain: LLMDomain
    text: str | None = None
    prompt: str | None = None
    response: str | None = None
    conversations: list[dict[str, Any]] | None = None
    chosen: str | None = None
    rejected: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
