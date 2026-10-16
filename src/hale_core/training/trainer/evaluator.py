"""``Evaluator``: Evaluator (Variant-agnostic training loop resolved from registries)."""

from __future__ import annotations

from typing import Protocol


class Evaluator(Protocol):
    def run(self, model, loader) -> dict[str, float]: ...
