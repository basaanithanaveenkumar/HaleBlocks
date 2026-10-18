"""Feed-forward registry: ``mlp``, ``geglu`` and ``moe`` blocks built by name."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from hale_core.nn.layers.ffn.mlp import DenseMLP, GeGLU
from hale_core.nn.layers.ffn.moe import DeepseekMoE
from hale_core.registry import NamedRegistry

FFN = NamedRegistry("ffn")


def register_ffn(name: str) -> Callable[[type], type]:
    """Decorator registering a feed-forward implementation under ``name``."""

    def deco(cls: type) -> type:
        FFN.add(name, cls)
        return cls

    return deco


register_ffn("mlp")(DenseMLP)
register_ffn("geglu")(GeGLU)
register_ffn("moe")(DeepseekMoE)


def _build_dense(cls: type, d_model: int, d_ff: int, dropout: float, **_: Any) -> Any:
    return cls(d_model, d_ff, dropout)


def _build_moe(
    cls: type,
    d_model: int,
    d_ff: int,
    dropout: float,
    *,
    moe_num_experts: int,
    moe_top_k: int,
    moe_num_shared: int,
) -> Any:
    return cls(
        d_model,
        d_ff,
        n_experts=moe_num_experts,
        top_k=moe_top_k,
        n_shared=moe_num_shared,
        dropout=dropout,
    )


# Strategy per registry key mapping the flat config onto the class constructor.
FFN_CONSTRUCTORS: dict[str, Callable[..., Any]] = {
    "mlp": _build_dense,
    "geglu": _build_dense,
    "moe": _build_moe,
}


def build_ffn(
    kind: str,
    *,
    d_model: int,
    d_ff: int,
    dropout: float = 0.0,
    moe_num_experts: int = 4,
    moe_top_k: int = 2,
    moe_num_shared: int = 1,
) -> Any:
    """Build the feed-forward block registered as ``kind``.

    Args:
        kind: registry key (``mlp``, ``geglu``, ``moe``).
        d_model: model width.
        d_ff: hidden width.
        dropout: dropout probability.
        moe_num_experts: routed experts (``moe`` only).
        moe_top_k: experts per token (``moe`` only).
        moe_num_shared: always-on shared experts (``moe`` only).

    Returns:
        The feed-forward module.

    Raises:
        KeyError: if ``kind`` is not registered.
    """
    cls = FFN.get(kind)
    construct = FFN_CONSTRUCTORS.get(kind, _build_dense)
    return construct(
        cls,
        d_model,
        d_ff,
        dropout,
        moe_num_experts=moe_num_experts,
        moe_top_k=moe_top_k,
        moe_num_shared=moe_num_shared,
    )
