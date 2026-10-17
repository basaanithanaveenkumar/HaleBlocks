"""Expose a :class:`CalibrationSource` through ONNX Runtime's ``CalibrationDataReader`` protocol."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from hale_core.deploy.calibration.calibration_source import CalibrationSource
from hale_core.deploy.core.io_signature import IOSignature


class OnnxCalibrationReader:
    """``get_next()`` → next ``{input: ndarray}`` (``None`` when exhausted); ``rewind()`` restarts.

    ORT quantization, AMD Quark and Olive consume this protocol. The class is registered as a
    *virtual* subclass of ``onnxruntime.quantization.CalibrationDataReader`` by
    :meth:`register_with_onnxruntime`, so ``isinstance`` checks pass without importing
    onnxruntime at module load.
    """

    def __init__(self, source: CalibrationSource, signature: IOSignature) -> None:
        self.source = source
        self.signature = signature
        self._iterator: Iterator[dict[str, np.ndarray]] = source.as_numpy(signature)

    def get_next(self) -> dict[str, np.ndarray] | None:
        """Next calibration batch for ONNX Runtime, or ``None`` when exhausted."""
        return next(self._iterator, None)

    def rewind(self) -> None:
        """Restart the batch stream from the beginning."""
        self._iterator = self.source.as_numpy(self.signature)

    @classmethod
    def register_with_onnxruntime(cls) -> None:
        """Register this class as a virtual subclass of ORT's ``CalibrationDataReader``."""
        from onnxruntime.quantization import CalibrationDataReader

        CalibrationDataReader.register(cls)
