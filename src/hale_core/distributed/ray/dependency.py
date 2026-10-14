"""Import Ray lazily with an actionable error message."""

from __future__ import annotations

from types import ModuleType


def require_ray() -> tuple[ModuleType, ModuleType, ModuleType]:
    """Return ``(ray, ray.train, ray.train.torch)`` or raise an ``ImportError`` explaining the fix.

    Ray is an optional dependency: it is only needed when ``train.ray.enabled`` is true.
    """
    try:
        import ray
        import ray.train
        import ray.train.torch
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "train.ray.enabled=true needs Ray Train: pip install 'hale-blocks[ray]' "
            "(or pip install 'ray[train]')"
        ) from exc
    return ray, ray.train, ray.train.torch
