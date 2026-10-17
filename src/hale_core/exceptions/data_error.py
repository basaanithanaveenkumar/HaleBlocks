"""A dataset cannot be converted, encoded or decoded."""

from __future__ import annotations

from hale_core.exceptions.hale_core_error import HaleCoreError


class DataError(HaleCoreError, ValueError):
    """Bad dataset content: an empty conversion, an unknown codec version, an invalid field name.

    Also a :class:`ValueError`, the built-in these sites raised before the hierarchy existed.
    """
