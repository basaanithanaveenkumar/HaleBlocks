"""Source of representative inputs for post-training quantization (Strategy pattern)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator

import numpy as np
import torch

from hale_core.deploy.core.io_signature import IOSignature


class CalibrationSource(ABC):
    """Yields ``{input_name: tensor}`` batches matching an :class:`IOSignature`.

    Quantizers differ in what they consume (ModelOpt: ``.npz``; Quark / ORT: a
    ``CalibrationDataReader``; QNN / ExecuTorch: forward calls; AI Hub: lists of arrays). They
    all start from these batches; :meth:`as_numpy` and :meth:`stacked` provide the common forms.
    """

    def __init__(self, num_batches: int = 8) -> None:
        if num_batches < 1:
            raise ValueError("num_batches must be >= 1")
        self.num_batches = num_batches

    @abstractmethod
    def batches(self, signature: IOSignature) -> Iterator[dict[str, torch.Tensor]]:
        """Yield ``num_batches`` input dicts."""

    def as_numpy(self, signature: IOSignature) -> Iterator[dict[str, np.ndarray]]:
        for batch in self.batches(signature):
            yield {k: v.detach().cpu().numpy() for k, v in batch.items()}

    def stacked(self, signature: IOSignature) -> dict[str, np.ndarray]:
        """All batches concatenated along axis 0 (the ``.npz`` layout ModelOpt expects)."""
        columns: dict[str, list[np.ndarray]] = {}
        for batch in self.as_numpy(signature):
            for name, array in batch.items():
                columns.setdefault(name, []).append(array)
        return {name: np.concatenate(arrays, axis=0) for name, arrays in columns.items()}
