"""Backward-compatible alias: the registry now lives in :mod:`hale_core.registry.registry`."""

from hale_core.registry.registry import BuildFunc, Registry, build_from_cfg

__all__ = ["BuildFunc", "Registry", "build_from_cfg"]
