"""Attention registry: ``mha``, ``gqa`` and ``mqa`` implementations built by name."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from hale_core.nn.layers.attention.grouped import GroupedQueryAttention
from hale_core.nn.layers.attention.mha import TorchMultiHeadAttention
from hale_core.registry import NamedRegistry

ATTENTION = NamedRegistry("attention")


def register_attention(name: str) -> Callable[[type], type]:
    """Decorator registering an attention implementation under ``name``."""

    def deco(cls: type) -> type:
        ATTENTION.add(name, cls)
        return cls

    return deco


register_attention("mha")(TorchMultiHeadAttention)
register_attention("gqa")(GroupedQueryAttention)
register_attention("mqa")(GroupedQueryAttention)


def _build_mha(cls: type, d_model: int, n_heads: int, dropout: float, **_: Any) -> Any:
    return cls(d_model, n_heads, dropout)


def _build_mqa(cls: type, d_model: int, n_heads: int, dropout: float, **extra: Any) -> Any:
    extra.pop("n_kv_heads", None)
    return cls(d_model, n_heads, n_kv_heads=1, dropout=dropout, **extra)


def _build_gqa(cls: type, d_model: int, n_heads: int, dropout: float, **extra: Any) -> Any:
    n_kv_heads = extra.pop("n_kv_heads", None)
    kv = n_kv_heads if n_kv_heads is not None else max(1, n_heads // 4)
    return cls(d_model, n_heads, n_kv_heads=kv, dropout=dropout, **extra)


# Strategy per registry key: how a flat config maps onto that class's constructor. A new
# attention type registers its class and, if its signature differs, a strategy here.
ATTENTION_CONSTRUCTORS: dict[str, Callable[..., Any]] = {
    "mha": _build_mha,
    "mqa": _build_mqa,
    "gqa": _build_gqa,
}


def build_attention(
    impl: str,
    *,
    d_model: int,
    n_heads: int,
    dropout: float = 0.0,
    n_kv_heads: int | None = None,
    rope_theta: float | None = None,
    rotary_dim: int | None = None,
    qk_norm: bool = False,
) -> TorchMultiHeadAttention | GroupedQueryAttention:
    """Build the attention module registered as ``impl``.

    Args:
        impl: registry key (``mha``, ``gqa``, ``mqa``).
        d_model: model width.
        n_heads: query heads.
        dropout: attention dropout.
        n_kv_heads: key/value heads for grouped-query attention.
        rope_theta: RoPE base; ``None`` disables rotary embeddings.
        rotary_dim: number of rotated dimensions per head (``None`` = full head).
        qk_norm: normalise queries and keys before the dot product.

    Returns:
        The attention module.

    Raises:
        KeyError: if ``impl`` is not registered.
    """
    cls = ATTENTION.get(impl)
    construct = ATTENTION_CONSTRUCTORS.get(impl, _build_gqa)
    return construct(
        cls,
        d_model,
        n_heads,
        dropout,
        n_kv_heads=n_kv_heads,
        rope_theta=rope_theta,
        rotary_dim=rotary_dim,
        qk_norm=qk_norm,
    )
