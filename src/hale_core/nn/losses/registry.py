"""Registered per-token losses. Add a file + `@register_token_loss` — callers stay closed."""

from __future__ import annotations

from collections.abc import Callable

from hale_core.registry import NamedRegistry

TOKEN_LOSSES = NamedRegistry("token_loss")


def register_token_loss(name: str):
    """Decorator registering a token-loss function under ``name``."""

    def deco(fn: Callable) -> Callable:
        TOKEN_LOSSES.add(name, fn)
        return fn

    return deco


def get_token_loss(name: str) -> Callable:
    """The token-loss function registered as ``name``."""
    return TOKEN_LOSSES.get(name)
