"""A graph contains operators a backend converter does not support."""

from __future__ import annotations

from hale_core.deploy.exceptions.deploy_error import DeployError


class ConversionError(DeployError):
    """Attributes:
    unsupported: every unsupported operator found (reported all at once, not one by one).
    """

    def __init__(self, message: str, unsupported: list[str] | None = None) -> None:
        self.unsupported = sorted(set(unsupported or []))
        super().__init__(
            message + (f"; unsupported ops: {self.unsupported}" if self.unsupported else "")
        )
