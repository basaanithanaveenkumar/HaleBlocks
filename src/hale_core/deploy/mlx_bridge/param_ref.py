"""Reference to a weight / buffer / constant stored in the program's parameter table."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ParamRef:
    """Resolved from :attr:`MLXProgram.params` (weights live outside the per-call environment)."""

    key: str
