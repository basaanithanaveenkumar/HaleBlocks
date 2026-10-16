"""Root of the HaleBlocks exception hierarchy."""

from __future__ import annotations


class HaleCoreError(Exception):
    """Base class of every error raised on purpose by :mod:`hale_core` and the Hale* packages.

    Downstream packages root their own hierarchies here (``DeployError``, ``HaleVLMError``, …),
    so a CLI or a job runner can separate expected, actionable failures (``except
    HaleCoreError``) from programming errors without listing every subclass.
    """
