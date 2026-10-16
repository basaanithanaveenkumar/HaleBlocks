"""An optional vendor SDK (TensorRT, MLX, MIGraphX, QNN ...) is not importable."""

from __future__ import annotations

from hale_core.deploy.exceptions.deploy_error import DeployError


class MissingDependencyError(DeployError, ImportError):
    """Raised when a stage needs a package that is not installed.

    Attributes:
        package: import name that failed (e.g. ``"tensorrt"``).
        hint: how to install it on a supported platform.
    """

    def __init__(self, package: str, hint: str = "") -> None:
        self.package = package
        self.hint = hint
        message = f"optional dependency {package!r} is not installed"
        super().__init__(f"{message}; {hint}" if hint else message)
