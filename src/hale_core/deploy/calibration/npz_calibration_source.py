"""Calibration batches stored as ``.npz`` files (one batch per file)."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from hale_core.deploy.calibration.calibration_source import CalibrationSource
from hale_core.deploy.core.io_signature import IOSignature
from hale_core.deploy.exceptions import ConfigError
from hale_core.deploy.registry import CALIBRATION


@CALIBRATION.register_module()
class NpzCalibrationSource(CalibrationSource):
    """Read ``directory/*.npz`` in sorted order; every file holds one array per input name."""

    def __init__(self, directory: str, num_batches: int = 64) -> None:
        super().__init__(num_batches)
        self.directory = Path(directory)

    def batches(self, signature: IOSignature) -> Iterator[dict[str, torch.Tensor]]:
        files = sorted(self.directory.glob("*.npz"))[: self.num_batches]
        if not files:
            raise ConfigError(f"no .npz calibration files in {self.directory}")
        for file in files:
            with np.load(file) as data:
                missing = [n for n in signature.input_names if n not in data]
                if missing:
                    raise ConfigError(f"{file} lacks calibration inputs {missing}")
                yield {
                    name: torch.from_numpy(np.array(data[name])) for name in signature.input_names
                }
