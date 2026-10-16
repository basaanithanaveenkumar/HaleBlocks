"""``DataModule``: DataModule (Variant-agnostic training loop resolved from registries)."""

from __future__ import annotations

from typing import Any, Protocol


class DataModule(Protocol):
    tokenizer: Any

    def train_loader(self): ...

    def val_loader(self, train_loader): ...

    def viz_prompt(self, step: int) -> str: ...
