"""Recursive dict merge with mmengine's ``_delete_`` semantics."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

DELETE_KEY = "_delete_"


class DictMerger:
    """Merge a child config onto its base.

    Rules (applied recursively):
        * mapping + mapping → merged key by key;
        * a child mapping containing ``_delete_: true`` **replaces** the base mapping instead;
        * anything else (scalars, lists) → the child value wins.

    Lists are replaced, not concatenated: a pipeline override must be able to drop a stage.
    Inputs are never mutated; the result is a deep copy.
    """

    def merge(self, base: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
        """Merge ``child`` onto ``base`` (see the class docstring for the rules).

        Args:
            base: parent mapping.
            child: overriding mapping; ``_delete_: true`` in a sub-mapping replaces instead of merging.

        Returns:
            A new deep-copied dict; neither input is mutated.
        """
        out = copy.deepcopy(dict(base))
        for key, value in child.items():
            if isinstance(value, Mapping):
                value = dict(value)
                replace = bool(value.pop(DELETE_KEY, False))
                if not replace and isinstance(out.get(key), Mapping):
                    out[key] = self.merge(out[key], value)
                    continue
                out[key] = self.strip(value)
            else:
                out[key] = copy.deepcopy(value)
        return out

    def strip(self, value: Any) -> Any:
        """Remove leftover ``_delete_`` markers (they mean nothing without a base)."""
        if isinstance(value, Mapping):
            return {k: self.strip(v) for k, v in value.items() if k != DELETE_KEY}
        if isinstance(value, list):
            return [self.strip(v) for v in value]
        return copy.deepcopy(value)
