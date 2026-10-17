"""``DataModule``: Structural type of the trainer's data source."""

from __future__ import annotations

from typing import Any, Protocol


class DataModule(Protocol):
    """Structural type of the trainer's data source."""
    tokenizer: Any

    def train_loader(self):
        """Iterable of training batches."""
        ...

    def val_loader(self, train_loader):
        """Iterable of validation batches (may reuse ``train_loader``)."""
        ...

    def viz_prompt(self, step: int) -> str:
        """Prompt or label used for visualisations at ``step``."""
        ...
