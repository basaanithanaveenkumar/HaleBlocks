"""Import ``webdataset`` lazily with an actionable error message."""

from __future__ import annotations

from types import ModuleType


def require_webdataset() -> ModuleType:
    """Return the ``webdataset`` module or raise an ``ImportError`` explaining the fix."""
    try:
        import webdataset
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "data.webdataset.enabled=true needs webdataset: pip install 'hale-blocks[webdataset]' "
            "(or pip install webdataset)"
        ) from exc
    return webdataset
