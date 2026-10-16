"""
DINOv2 Vision Transformer Backbone
=====================================

Paper: "DINOv2: Learning Robust Visual Features without Supervision"
       (Oquab et al., 2023) — https://arxiv.org/abs/2304.07193

Architecture Overview
---------------------
DINOv2 builds on ViT with three key improvements for self-supervised training:

1. Register Tokens
   Large ViTs develop high-norm artifact tokens in background regions.  DINOv2
   introduces n_register learnable tokens inserted AFTER the CLS token.  Registers
   absorb global image statistics (background, repetitive textures), freeing patch
   tokens to carry semantically meaningful local features.  At inference they are
   discarded — only CLS and patch tokens form the output.

2. LayerScale (Caron et al., 2021 — CaiT)
   A per-channel learnable scalar γ initialised at a small value (1e-5 for large
   models) multiplied onto each residual output:
       x = x + γ · Attention(LN(x))
   This prevents deep networks from exploding gradients during early training by
   making residuals near-zero initially, then letting γ grow naturally.

3. Flash Attention (F.scaled_dot_product_attention)
   Uses PyTorch 2.0's fused kernel when available.  Falls back to manual attention
   on older versions.  No numerical difference — the fused kernel is just faster.

Variants registered
-------------------
  dino_vit_small  — 384-dim, 6 heads,  12 layers, patch=14, 4 registers
  dino_vit_base   — 768-dim, 12 heads, 12 layers, patch=14, 4 registers
  dino_vit_large  — 1024-dim, 16 heads, 24 layers, patch=14, 4 registers
  dino_vit_giant  — 1536-dim, 24 heads, 40 layers, patch=14, 4 registers  ← DINOv2-g
"""

from hale_core.nn.vision.vit.dino.din_ov2_backbone import DINOv2Backbone
from hale_core.nn.vision.vit.dino.dino_attention import DINOAttention
from hale_core.nn.vision.vit.dino.dino_block import DINOBlock
from hale_core.nn.vision.vit.dino.dino_vi_t_base import DINOViTBase
from hale_core.nn.vision.vit.dino.dino_vi_t_giant import DINOViTGiant
from hale_core.nn.vision.vit.dino.dino_vi_t_large import DINOViTLarge
from hale_core.nn.vision.vit.dino.dino_vi_t_small import DINOViTSmall
from hale_core.nn.vision.vit.dino.layer_scale import LayerScale

__all__ = [
    "DINOAttention",
    "DINOBlock",
    "DINOViTBase",
    "DINOViTGiant",
    "DINOViTLarge",
    "DINOViTSmall",
    "DINOv2Backbone",
    "LayerScale",
]
