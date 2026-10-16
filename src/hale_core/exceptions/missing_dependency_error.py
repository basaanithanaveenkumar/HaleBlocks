"""An optional dependency (vendor SDK, extra) needed by the requested feature is not installed."""

from __future__ import annotations

from hale_core.exceptions.hale_core_error import HaleCoreError


class MissingDependencyError(HaleCoreError, ImportError):
    """Raised instead of a bare :class:`ImportError` when an optional package is missing.

    Attributes:
        package: import name that failed (e.g. ``"tensorrt"``).
        hint: how to install it (e.g. ``pip install 'hale-blocks[deploy-nvidia]'``).

    Args:
        package: import name that failed.
        hint: installation hint appended to the message.
    """

    def __init__(self, package: str, hint: str = "") -> None:
        self.package = package
        self.hint = hint
        message = f"optional dependency {package!r} is not installed"
        super().__init__(f"{message}; {hint}" if hint else message)
