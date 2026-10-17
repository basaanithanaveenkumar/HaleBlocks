"""A model could not be exported or compiled for a runtime."""

from __future__ import annotations

from hale_core.exceptions.hale_core_error import HaleCoreError


class ExportError(HaleCoreError, RuntimeError):
    """An exporter (TensorRT, ONNX …) failed to parse, build or load an artifact.

    Also a :class:`RuntimeError`, the built-in these sites raised before the hierarchy existed.
    """
