"""``Evaluator``: Structural type of the trainer's evaluator."""

from __future__ import annotations

from typing import Protocol


class Evaluator(Protocol):
    """Structural type of the trainer's evaluator."""

    def run(self, model, loader) -> dict[str, float]:
        """Evaluate ``model`` on ``loader`` and return metric name -> value."""
        ...
