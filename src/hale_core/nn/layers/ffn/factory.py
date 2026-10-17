"""Feed-forward registry: ``mlp``, ``geglu`` and ``moe`` blocks built by name."""

from __future__ import annotations

from hale_core.nn.layers.ffn.mlp import DenseMLP, GeGLU
from hale_core.nn.layers.ffn.moe import DeepseekMoE
from hale_core.registry import NamedRegistry

FFN = NamedRegistry("ffn")


def register_ffn(name: str):
    """Decorator registering a feed-forward implementation under ``name``."""
    def deco(cls):
        FFN.add(name, cls)
        return cls

    return deco


register_ffn("mlp")(DenseMLP)
register_ffn("geglu")(GeGLU)
register_ffn("moe")(DeepseekMoE)


def build_ffn(
    kind: str,
    *,
    d_model: int,
    d_ff: int,
    dropout: float = 0.0,
    moe_num_experts: int = 4,
    moe_top_k: int = 2,
    moe_num_shared: int = 1,
):
    """Build the feed-forward block registered as ``kind``.

    Args:
        kind: registry key (``mlp``, ``geglu``, ``moe``).
        d_model: model width.
        d_ff: hidden width.
        dropout: dropout probability.
        moe_num_experts: routed experts (``moe`` only).
        moe_top_k: experts per token (``moe`` only).
        moe_num_shared: always-on shared experts (``moe`` only).
    """
    if kind == "mlp":
        return FFN.get("mlp")(d_model, d_ff, dropout)
    if kind == "geglu":
        return FFN.get("geglu")(d_model, d_ff, dropout)
    if kind == "moe":
        return FFN.get("moe")(
            d_model,
            d_ff,
            n_experts=moe_num_experts,
            top_k=moe_top_k,
            n_shared=moe_num_shared,
            dropout=dropout,
        )
    raise KeyError(f"unknown ffn {kind!r}; registered={sorted(FFN.items)}")
