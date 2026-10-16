"""Reference to a value computed by an earlier graph node (or a program input)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValueRef:
    """Placeholder in a node's argument tree, resolved from the execution environment."""

    name: str
