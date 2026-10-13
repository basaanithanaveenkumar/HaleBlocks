"""Hybrid (conv + attention) backbone family."""

from hale_core.nn.vision.hybrid import coatnet, maxvit  # noqa: F401 — triggers @register decorators

__all__ = ["coatnet", "maxvit"]
