"""``DatasetKind``.

Top-level dataset family (LLM text, VLM, locomotion …), used to route specs to loaders.
"""

from __future__ import annotations

from enum import StrEnum


class DatasetKind(StrEnum):
    """Top-level dataset family (LLM text, VLM, locomotion …), used to route specs to loaders."""

    LLM = "llm"
    VLM = "vlm"
    LOCOMOTION = "locomotion"
