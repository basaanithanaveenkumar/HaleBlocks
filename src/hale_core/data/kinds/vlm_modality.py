"""``VLMModality``: What a VLM sample contains (image-text, video-text, interleaved …)."""

from __future__ import annotations

from enum import StrEnum


class VLMModality(StrEnum):
    """What a VLM sample contains (image-text, video-text, interleaved …)."""

    TEXT = "text"
    IMAGE = "image"
    MULTI_IMAGE = "multi_image"
    VIDEO = "video"
    MIXED = "mixed"
