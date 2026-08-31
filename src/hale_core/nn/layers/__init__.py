"""Reusable neural primitives: attention, FFN, norms, layers. No vocab, no training loop."""

from hale_core.nn.layers.attention import (
    ATTENTION,
    build_attention,
    build_attn_mask,
    or_masks,
    register_attention,
    sliding_window_mask,
)
from hale_core.nn.layers.dit import DiTBlock, DiTFinalLayer
from hale_core.nn.layers.embeddings import SinusoidalTimeEmbedding
from hale_core.nn.layers.ffn import FFN, build_ffn, register_ffn
from hale_core.nn.layers.layer import TransformerLayer
from hale_core.nn.layers.norm import RMSNorm

__all__ = [
    "ATTENTION",
    "FFN",
    "RMSNorm",
    "SinusoidalTimeEmbedding",
    "TransformerLayer",
    "DiTBlock",
    "DiTFinalLayer",
    "build_attention",
    "build_attn_mask",
    "build_ffn",
    "or_masks",
    "register_attention",
    "register_ffn",
    "sliding_window_mask",
]
