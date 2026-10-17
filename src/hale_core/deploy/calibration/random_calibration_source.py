"""Seeded random inputs: shapes and value ranges from the signature."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from hale_core.deploy.calibration.calibration_source import CalibrationSource
from hale_core.deploy.core.io_signature import IOSignature
from hale_core.deploy.registry import CALIBRATION


@CALIBRATION.register_module()
class RandomCalibrationSource(CalibrationSource):
    """Good enough to exercise a quantization flow end to end; use real data for accuracy.

    Args:
        num_batches: number of batches.
        which: signature corner (``min``/``opt``/``max``) used for shapes.
        seed: generator seed (deterministic calibration → reproducible engines).
    """

    def __init__(self, num_batches: int = 8, which: str = "opt", seed: int = 0) -> None:
        super().__init__(num_batches)
        self.which = which
        self.seed = seed

    def batches(self, signature: IOSignature) -> Iterator[dict[str, torch.Tensor]]:
        """Yield random batches matching the signature (shapes at ``opt``, valid integer ranges)."""
        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(self.num_batches):
            yield signature.sample(self.which, generator)
