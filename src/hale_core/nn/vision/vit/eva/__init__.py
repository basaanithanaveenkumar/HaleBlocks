"""
EVA-02 Vision Transformer Backbone
=====================================

Paper: "EVA-02: A Visual Representation Powerhouse Reconstructed from EVA"
       (Fang et al., 2023) — https://arxiv.org/abs/2303.11331

Architecture Overview
---------------------
EVA-02 introduces three key modifications over plain ViT that together achieve
state-of-the-art visual representations:

1. Rotary Position Embedding (2D RoPE)
   Instead of learned absolute positional embeddings added to all tokens, RoPE
   encodes positions by rotating Q and K vectors with position-dependent angles.

   For a 1D sequence, RoPE rotates consecutive pairs (x_{2i}, x_{2i+1}) by θ:
       x' = x·cos(θ) + rotate_half(x)·sin(θ)
   where rotate_half swaps pairs with a sign flip: [a, b, c, d] → [-b, a, -d, c].

   For 2D images, EVA-02 uses separate RoPE for rows and columns:
       - Each head's d/2 dimensions are split into d/4 row + d/4 col dims
       - Row dims rotate by position_y * θ_i, col dims by position_x * θ_i
   This generalises to arbitrary resolutions without interpolation artefacts,
   unlike absolute learned positional embeddings.

2. SwiGLU FFN
   Standard ViT uses GELU(Linear(x)) as the activation gate.  SwiGLU uses:
       SwiGLU(x) = SiLU(W₁x) ⊙ W₂x   (element-wise gate)
   This is a gated linear unit (GLU) variant shown to outperform GELU MLPs at
   the same parameter count (Noam Shazeer, 2020: "GLU Variants Improve Transformer").
   The intermediate dim is scaled by 2/3 × 4 = 8/3 ≈ 2.667× to keep parameter
   count equal to a standard 4× MLP (two matrices instead of one gate).

3. Sub-Layer Normalization (Sub-LN) for Q and K
   Applying LayerNorm to Q and K before computing attention stabilises training
   by preventing attention logit explosion at large scale (large dim, many heads).
   This is independent from the pre-norm applied to the full token sequence.

Output:
   EVA-02 uses mean-pooling over all patch tokens (no CLS token in EVA-02 style)
   as the global pooled_output.

Variants registered
-------------------
  eva_02_small — 384-dim, 6 heads,  12 layers, patch=16
  eva_02_base  — 768-dim, 12 heads, 12 layers, patch=16
  eva_02_large — 1024-dim, 16 heads, 24 layers, patch=14
"""

import hale_core.nn.vision.vit.eva._helpers  # noqa: F401 - module-level registrations
from hale_core.nn.vision.vit.eva.eva02_backbone import EVA02Backbone
from hale_core.nn.vision.vit.eva.eva02_base import EVA02Base
from hale_core.nn.vision.vit.eva.eva02_large import EVA02Large
from hale_core.nn.vision.vit.eva.eva02_small import EVA02Small
from hale_core.nn.vision.vit.eva.eva_attention import EVAAttention
from hale_core.nn.vision.vit.eva.eva_block import EVABlock
from hale_core.nn.vision.vit.eva.swi_gluffn import SwiGLUFFN

__all__ = [
    "EVA02Backbone",
    "EVA02Base",
    "EVA02Large",
    "EVA02Small",
    "EVAAttention",
    "EVABlock",
    "SwiGLUFFN",
]
