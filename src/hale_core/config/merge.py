"""Nested dict merge used by YAML inheritance (thin wrapper over :class:`DictMerger`)."""

from __future__ import annotations

from typing import Any

from hale_core.config.dict_merger import DictMerger

_INHERITANCE_KEYS = ("inherits", "_base_")


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Merge ``override`` onto ``base`` recursively; the override wins on conflicts.

    Inheritance keys (``inherits``, ``_base_``) in ``override`` are dropped, and a mapping that
    contains ``_delete_: true`` replaces the base mapping instead of merging into it.
    Neither input is mutated.

    Args:
        base: the parent mapping.
        override: the child mapping.

    Returns:
        A new merged dict.
    """
    child = {k: v for k, v in override.items() if k not in _INHERITANCE_KEYS}
    return DictMerger().merge(base, child)
