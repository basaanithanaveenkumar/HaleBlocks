"""``VLMModality``: VLMModality (Shared dataset kind and stage enums)."""

from __future__ import annotations

from enum import StrEnum


class VLMModality(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    MULTI_IMAGE = "multi_image"
    VIDEO = "video"
    MIXED = "mixed"
