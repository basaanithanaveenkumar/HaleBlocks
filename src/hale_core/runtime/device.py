"""Device selection: an explicit request if it is usable, else CUDA, then MPS, then CPU."""

from __future__ import annotations

import torch
from loguru import logger


class DeviceResolver:
    """Picks the torch device string for a run."""

    def resolve(self, device: str | None = None) -> str:
        """The device to use.

        Args:
            device: requested device; an unavailable ``cuda``/``mps`` request falls back (with a
                warning) to automatic selection.

        Returns:
            ``"cuda"``, ``"mps"``, ``"cpu"`` or the requested device string.
        """
        if device:
            logger.debug("using requested device={}", device)
            if device == "cuda" and not torch.cuda.is_available():
                # The warning always promised a fallback; returning "cuda" here only moved the
                # failure to the first ``.to("cuda")``.
                logger.warning("cuda requested but not available; falling back")
                return self.resolve(None)
            if device == "mps" and not (
                hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            ):
                logger.warning("mps requested but not available; falling back to cpu")
                return "cpu"
            return device
        if torch.cuda.is_available():
            logger.info("auto-selected device=cuda")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("auto-selected device=mps")
            return "mps"
        logger.info("auto-selected device=cpu")
        return "cpu"


_RESOLVER = DeviceResolver()


def get_device(device: str | None = None) -> str:
    """Resolve ``device`` with the shared :class:`DeviceResolver`."""
    return _RESOLVER.resolve(device)
