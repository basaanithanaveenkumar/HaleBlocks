"""``DatasetKind``: DatasetKind (Shared dataset kind and stage enums)."""

from __future__ import annotations

from enum import StrEnum


class DatasetKind(StrEnum):
    LLM = "llm"
    VLM = "vlm"
    LOCOMOTION = "locomotion"
