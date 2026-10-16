"""A pipeline stage failed while producing its artifacts."""

from __future__ import annotations

from hale_core.deploy.exceptions.deploy_error import DeployError


class StageError(DeployError):
    """Wraps the original exception of a failing stage with the stage name.

    Attributes:
        stage: name of the stage that failed.
    """

    def __init__(self, stage: str, message: str) -> None:
        self.stage = stage
        super().__init__(f"[{stage}] {message}")
