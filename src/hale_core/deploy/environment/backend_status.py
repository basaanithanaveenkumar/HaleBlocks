"""Availability of one deployment backend on the current machine."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BackendStatus:
    """Whether one deployment backend (vendor, name) is usable here, and why not."""

    vendor: str
    backend: str
    available: bool
    version: str | None
    detail: str
