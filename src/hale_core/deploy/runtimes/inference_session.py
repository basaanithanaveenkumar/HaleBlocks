"""Uniform ``run(named numpy inputs) -> list of numpy outputs`` over every backend runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import ClassVar

import numpy as np

from hale_core.deploy.core.optional_dependency import OptionalDependency


class InferenceSession(ABC):
    """Strategy interface used by :class:`RuntimeParityStage` (and handy for smoke tests).

    Each implementation loads one artifact kind (ONNX file, TensorRT engine, ``.mlxfn``,
    MIGraphX program ...) and runs it on numpy inputs, so parity checking is written once.
    """

    dependencies: ClassVar[tuple[OptionalDependency, ...]] = ()

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        for dep in self.dependencies:
            dep.load()  # raises MissingDependencyError with an install hint

    @abstractmethod
    def run(self, inputs: Mapping[str, np.ndarray]) -> list[np.ndarray]:
        """Run once; outputs are returned in the model's output order."""
