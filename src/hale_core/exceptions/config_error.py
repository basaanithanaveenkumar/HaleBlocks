"""Invalid, missing or circular configuration."""

from __future__ import annotations

from hale_core.exceptions.hale_core_error import HaleCoreError


class ConfigError(HaleCoreError, ValueError):
    """A config file cannot be read, inherits in a cycle, has a bad override or fails validation.

    Also a :class:`ValueError`, so call sites that predate the hierarchy keep working.
    """
