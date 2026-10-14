"""A picklable ``IterableDataset`` with a known length."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from torch.utils.data import IterableDataset


class SizedStream(IterableDataset):
    """Wrap any iterable pipeline and report ``length`` from ``__len__``.

    ``webdataset``'s own ``.with_length()`` creates a class on the fly, which cannot be pickled -
    DataLoader workers started with ``spawn`` (the default on macOS and Windows, and common under
    Ray) then fail. This plain class pickles fine and keeps ``len(DataLoader)`` working for
    trainers that size their LR schedule from it.

    Args:
        pipeline: the iterable to stream (re-iterated for every epoch).
        length: samples per epoch *for this rank* (what ``__len__`` reports).
    """

    def __init__(self, pipeline: Iterable[Any], length: int) -> None:
        super().__init__()
        self.pipeline = pipeline
        self.length = length

    def __iter__(self) -> Iterator[Any]:
        return iter(self.pipeline)

    def __len__(self) -> int:
        return self.length
