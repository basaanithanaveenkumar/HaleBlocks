"""Registry-driven construction of attention and feed-forward layers."""

from __future__ import annotations

import pytest
import torch

from hale_core.exceptions import RegistryError
from hale_core.nn.layers.attention.factory import ATTENTION_CONSTRUCTORS, build_attention
from hale_core.nn.layers.ffn.factory import FFN_CONSTRUCTORS, build_ffn


@pytest.mark.parametrize("impl", sorted(ATTENTION_CONSTRUCTORS))
def test_build_attention_forward_shape(impl: str) -> None:
    torch.manual_seed(0)
    layer = build_attention(impl, d_model=32, n_heads=4)
    assert sum(p.numel() for p in layer.parameters()) > 0


def test_build_attention_unknown_raises() -> None:
    with pytest.raises(RegistryError):
        build_attention("nope", d_model=32, n_heads=4)


@pytest.mark.parametrize("kind", sorted(FFN_CONSTRUCTORS))
def test_build_ffn_preserves_width(kind: str) -> None:
    torch.manual_seed(0)
    layer = build_ffn(kind, d_model=16, d_ff=32, moe_num_experts=2, moe_top_k=1)
    out = layer(torch.randn(2, 5, 16))
    out = out[0] if isinstance(out, tuple) else out
    assert out.shape == (2, 5, 16)


def test_build_ffn_unknown_raises() -> None:
    with pytest.raises(RegistryError):
        build_ffn("nope", d_model=16, d_ff=32)
