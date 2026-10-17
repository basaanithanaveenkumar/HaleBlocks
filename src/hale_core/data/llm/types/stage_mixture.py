"""``StageMixture``: StageMixture (LLM dataset sample skeleton (SmolLM2 paper))."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StageMixture:
    """A training stage's dataset mixture: ``(dataset name, weight)`` pairs."""
    stage: int
    datasets: tuple[tuple[str, float], ...]
    description: str = ""
