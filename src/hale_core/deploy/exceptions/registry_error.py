"""Registry lookups and registrations that cannot be satisfied."""

from __future__ import annotations

from hale_core.deploy.exceptions.deploy_error import DeployError


class RegistryError(DeployError, KeyError):
    """Unknown or duplicate registry key, or a config dict without a ``type``.

    Also a :class:`KeyError` so ``except KeyError`` call sites keep working.
    """

    def __str__(self) -> str:  # KeyError quotes its message; keep it readable
        return str(self.args[0]) if self.args else ""
