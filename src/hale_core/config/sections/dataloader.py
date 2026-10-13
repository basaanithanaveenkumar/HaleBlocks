"""DataLoader sub-config: worker, memory, and batching settings."""

from __future__ import annotations

from hale_core.config.sections.common import StrictModel


class DataLoaderConfig(StrictModel):
    """torch.utils.data.DataLoader configuration.

    YAML example::

        data:
          loader:
            num_workers: 4
            pin_memory: true
            prefetch_factor: 2
            persistent_workers: true
            drop_last: true
    """

    num_workers: int = 0
    pin_memory: bool = False
    prefetch_factor: int | None = None
    persistent_workers: bool = False
    drop_last: bool = False
    shuffle: bool = True
    collator: str = "default"
