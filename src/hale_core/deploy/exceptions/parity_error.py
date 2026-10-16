"""An exported model's outputs drift from the PyTorch reference beyond tolerance."""

from __future__ import annotations

from hale_core.deploy.exceptions.deploy_error import DeployError


class ParityError(DeployError):
    """Numerical validation failed.

    Attributes:
        metrics: per-output error metrics that triggered the failure.
    """

    def __init__(self, message: str, metrics: dict | None = None) -> None:
        self.metrics = metrics or {}
        super().__init__(message)
