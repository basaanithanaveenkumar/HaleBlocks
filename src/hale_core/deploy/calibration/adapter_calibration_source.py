"""Calibration batches provided by the model adapter (real tokens, images, robot states)."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from hale_core.deploy.adapters.model_adapter import ModelAdapter
from hale_core.deploy.calibration.calibration_source import CalibrationSource
from hale_core.deploy.calibration.random_calibration_source import RandomCalibrationSource
from hale_core.deploy.core.io_signature import IOSignature
from hale_core.deploy.registry import CALIBRATION


@CALIBRATION.register_module()
class AdapterCalibrationSource(CalibrationSource):
    """Delegate to :meth:`ModelAdapter.calibration_batches`, falling back to random inputs.

    The adapter is injected by the runner (``default_args``), so configs only say
    ``calibration: {type: AdapterCalibrationSource, num_batches: 32}``.
    """

    def __init__(self, adapter: ModelAdapter, num_batches: int = 32, seed: int = 0) -> None:
        super().__init__(num_batches)
        self.adapter = adapter
        self.fallback = RandomCalibrationSource(num_batches, seed=seed)

    def batches(self, signature: IOSignature) -> Iterator[dict[str, torch.Tensor]]:
        provided = self.adapter.calibration_batches(self.num_batches)
        if provided is None:
            yield from self.fallback.batches(signature)
            return
        for i, batch in enumerate(provided):
            if i >= self.num_batches:
                break
            yield {name: batch[name] for name in signature.input_names}
