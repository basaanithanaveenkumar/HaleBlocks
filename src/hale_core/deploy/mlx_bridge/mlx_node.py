"""One operator call of a converted program."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MLXNode:
    """Attributes:
    name: output value name (the FX node name).
    op: registry key of the converter (``"aten.linear.default"``).
    fn: the converter callable.
    args / kwargs: argument trees with :class:`ValueRef` / :class:`ParamRef` leaves.
    release: value names whose last use is this node (freed right after it runs).
    """

    name: str
    op: str
    fn: Callable[..., Any]
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    release: tuple[str, ...] = field(default_factory=tuple)
