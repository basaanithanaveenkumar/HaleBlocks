"""``StageMixture``: StageMixture (LLM dataset sample skeleton (SmolLM2 paper))."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageMixture:
    stage: int
    datasets: tuple[tuple[str, float], ...]
    description: str = ""
