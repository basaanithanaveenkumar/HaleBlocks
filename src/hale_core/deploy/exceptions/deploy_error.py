"""Root of the deployment exception hierarchy."""

from __future__ import annotations

from hale_core.exceptions import HaleCoreError


class DeployError(HaleCoreError):
    """Base class of every error raised by :mod:`hale_core.deploy`.

    Catching ``DeployError`` lets a caller (CLI, trainer hook, CI job) separate deployment
    failures from programming errors without listing every subclass.
    """
