"""Neural building blocks: layers, stacks, backbones, losses, optimizers."""

from hale_core.nn.backbones import (
    BACKBONES,
    DiTBackbone,
    GPTBackbone,
    LGTBackbone,
    TransformerBackbone,
    build_backbone,
)
from hale_core.nn.layers import build_attention, build_attn_mask, build_ffn
from hale_core.nn.losses import TOKEN_LOSSES, get_token_loss, register_token_loss, token_nll
from hale_core.nn.optim import build_optimizer
from hale_core.nn.stacks import DiTStack, GPTStack, LGTStack

__all__ = [
    "BACKBONES",
    "build_backbone",
    "GPTBackbone",
    "LGTBackbone",
    "TransformerBackbone",
    "DiTBackbone",
    "GPTStack",
    "LGTStack",
    "DiTStack",
    "build_attn_mask",
    "build_attention",
    "build_ffn",
    "TOKEN_LOSSES",
    "register_token_loss",
    "get_token_loss",
    "token_nll",
    "build_optimizer",
]
