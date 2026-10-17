"""``LLMDomain``: LLMDomain (Shared dataset kind and stage enums)."""

from __future__ import annotations

from enum import StrEnum


class LLMDomain(StrEnum):
    """Content domain of an LLM dataset (web, code, math, chat …), used for mixtures."""
    WEB = "web"
    MATH = "math"
    CODE = "code"
    SYNTHETIC = "synthetic"
    INSTRUCTION = "instruction"
    PREFERENCE = "preference"
