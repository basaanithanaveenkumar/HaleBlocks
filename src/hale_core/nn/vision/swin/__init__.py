"""Swin transformer backbone family."""

from hale_core.nn.vision.swin import swinv2  # noqa: F401 — triggers @register decorators

__all__ = ["swinv2"]
