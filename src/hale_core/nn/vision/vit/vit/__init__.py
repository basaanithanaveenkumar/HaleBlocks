"""
Vision Transformer (ViT) Backbone
===================================

Original paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition
at Scale" (Dosovitskiy et al., 2020) — https://arxiv.org/abs/2010.11929

Architecture Overview
---------------------
ViT applies a standard Transformer encoder directly to sequences of flattened 2D
image patches, with minimal inductive bias.  The key insight is that patch sequences
can be treated exactly like word token sequences in NLP.

Pipeline:
    Image [B, 3, H, W]
       ↓  PatchEmbed (Conv2d stride=patch_size)
    Patches [B, N, D]
       ↓  prepend CLS token  +  add positional embedding
    Sequence [B, N+1, D]
       ↓  N×ViTBlock (LayerNorm → MHA → LayerNorm → MLP, pre-norm)
    [B, N+1, D]
       ↓  final LayerNorm
    pooled_output = x[:, 0]  (CLS token)
    patch_tokens  = x[:, 1:] (patch tokens, reshaped to [B, H/p, W/p, D])

Design choices carried forward
--------------------------------
- Pre-norm (LN before Q/K/V), NOT post-norm — more stable for large models
- Single combined in_proj (3×D → 3D), then split, for cache-friendly GEMM layout
- GELU activation in MLP (not ReLU)
- Learned 1D absolute positional embeddings (one vector per patch + CLS)
- Drop-path NOT included here — added in DeiT/DINOv2; keep base ViT minimal
- Dropout in attention (attn_drop) and in MLP (mlp_drop)

Variants registered
-------------------
  vit_tiny   — ViT-Ti/16:   192-dim, 3 heads, 12 layers
  vit_small  — ViT-S/16:    384-dim, 6 heads, 12 layers
  vit_base   — ViT-B/16:    768-dim, 12 heads, 12 layers
  vit_large  — ViT-L/16:   1024-dim, 16 heads, 24 layers
  vit_huge   — ViT-H/14:   1280-dim, 16 heads, 32 layers
  vit_22b    — ViT-22B/14: 6144-dim, 48 heads, 48 layers  (Zhai et al. 2022)
"""

from hale_core.nn.vision.vit.vit._helpers import (
    _trunc_normal_,  # noqa: F401 - _trunc_normal_ is imported by Halos-lift
)
from hale_core.nn.vision.vit.vit.patch_embed import PatchEmbed
from hale_core.nn.vision.vit.vit.vi_t22_b import ViT22B
from hale_core.nn.vision.vit.vit.vi_t_attention import ViTAttention
from hale_core.nn.vision.vit.vit.vi_t_backbone import ViTBackbone
from hale_core.nn.vision.vit.vit.vi_t_base import ViTBase
from hale_core.nn.vision.vit.vit.vi_t_block import ViTBlock
from hale_core.nn.vision.vit.vit.vi_t_huge import ViTHuge
from hale_core.nn.vision.vit.vit.vi_t_large import ViTLarge
from hale_core.nn.vision.vit.vit.vi_t_small import ViTSmall
from hale_core.nn.vision.vit.vit.vi_t_tiny import ViTTiny
from hale_core.nn.vision.vit.vit.vi_tmlp import ViTMLP

__all__ = [
    "PatchEmbed",
    "ViT22B",
    "ViTAttention",
    "ViTBackbone",
    "ViTBase",
    "ViTBlock",
    "ViTHuge",
    "ViTLarge",
    "ViTMLP",
    "ViTSmall",
    "ViTTiny",
]
