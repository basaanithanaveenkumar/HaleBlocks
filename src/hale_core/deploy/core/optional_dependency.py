"""Lazy, cached access to optional vendor packages (TensorRT, MLX, MIGraphX, QNN ...)."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from functools import cached_property
from types import ModuleType

from hale_core.deploy.exceptions import MissingDependencyError


@dataclass(frozen=True)
class OptionalDependency:
    """Describe an optional import and how to install it.

    ``available`` uses ``importlib.util.find_spec`` (no import side effects, cheap enough to call
    while validating a pipeline); ``load`` imports on first use and caches the module.
    """

    module: str
    hint: str = ""

    @cached_property
    def available(self) -> bool:
        """Whether the module can be imported (checked without importing it)."""
        try:
            return importlib.util.find_spec(self.module) is not None
        except (ImportError, ValueError):  # parent package missing / broken namespace package
            return False

    def load(self) -> ModuleType:
        """Import and return the module.

        Raises:
            MissingDependencyError: if it is not installed.
        """
        try:
            return importlib.import_module(self.module)
        except ImportError as exc:
            raise MissingDependencyError(self.module, self.hint) from exc
