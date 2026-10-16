"""A pipeline is mis-assembled (unsatisfiable artifact requirements, empty pipeline ...)."""

from __future__ import annotations

from hale_core.deploy.exceptions.deploy_error import DeployError


class PipelineError(DeployError):
    """Raised before any stage runs when the stage graph cannot be executed."""
