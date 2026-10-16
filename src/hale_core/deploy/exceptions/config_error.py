"""Invalid, missing or circular deployment configuration."""

from __future__ import annotations

from hale_core.deploy.exceptions.deploy_error import DeployError


class ConfigError(DeployError):
    """A config file cannot be read, inherits in a cycle, or fails schema validation."""
