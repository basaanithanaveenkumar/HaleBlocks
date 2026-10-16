"""Root of the deployment exception hierarchy."""

from __future__ import annotations


class DeployError(Exception):
    """Base class of every error raised by :mod:`hale_core.deploy`.

    Catching ``DeployError`` lets a caller (CLI, trainer hook, CI job) separate deployment
    failures from programming errors without listing every subclass.
    """
