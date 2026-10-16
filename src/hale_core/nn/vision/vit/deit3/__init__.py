"""
DeiT III Vision Transformer Backbone
======================================

Paper: "DeiT III: Revenge of the ViT" (Touvron et al., 2022)
       https://arxiv.org/abs/2204.07118

Architecture Overview
---------------------
DeiT III demonstrates that a vanilla ViT trained with the right data augmentation
and regularisation recipe (3-Augmentation, or 3-Reg) matches or beats more complex
architectures.  The architectural changes over baseline ViT are minimal:

1. LayerScale (Touvron et al., 2021 — CaiT)
   Same as DINOv2: per-channel learnable scale γ on each residual branch.
   Initial value: 1e-6 for large models (smaller than DINOv2's 1e-5 to handle
   more aggressive augmentation at the start of training).
   γ is a free parameter — at convergence it may be near 1.0 (full residual)
   or remain small (dampened contribution).

2. No CLS token — mean-pooling
   Unlike baseline ViT which uses a CLS token, DeiT III pools all patch tokens:
       pooled = mean(patch_tokens, dim=1)
   This removes the asymmetry between CLS and patch tokens, and all patches
   contribute equally to the global representation.  This simplifies transfer
   learning: any patch can be used for dense prediction without the CLS token
   occupying a "special" position.

3. Absolute learned positional embeddings (same as ViT)
   DeiT III does NOT use RoPE — it keeps the simple learned absolute embeddings.
   The 3-Reg augmentation (Random Resize Crop, RandAugment, Random Erasing) is
   the key to the strong performance, not architecture changes.

3-Augmentation training recipe (not implemented here, documented for reference)
  - RandomResizedCrop(224)
  - RandAugment(n=2, m=9) — two magnitude-9 transformations per image
  - RandomErasing(prob=0.25, scale=(0.02, 0.33))
  - Mixup(alpha=0.8) + CutMix(alpha=1.0)
  - Label smoothing (ε=0.1)
  - Repeated augmentation (same image augmented twice per batch for diversity)
  This recipe is applied in the training pipeline, not the backbone.

Variants registered
-------------------
  deit3_small — 384-dim, 6 heads,  12 layers, patch=16
  deit3_base  — 768-dim, 12 heads, 12 layers, patch=16
  deit3_large — 1024-dim, 16 heads, 24 layers, patch=16
  deit3_huge  — 1280-dim, 16 heads, 32 layers, patch=14
"""

from hale_core.nn.vision.vit.deit3.dei_t3_attention import DeiT3Attention
from hale_core.nn.vision.vit.deit3.dei_t3_backbone import DeiT3Backbone
from hale_core.nn.vision.vit.deit3.dei_t3_base import DeiT3Base
from hale_core.nn.vision.vit.deit3.dei_t3_block import DeiT3Block
from hale_core.nn.vision.vit.deit3.dei_t3_huge import DeiT3Huge
from hale_core.nn.vision.vit.deit3.dei_t3_large import DeiT3Large
from hale_core.nn.vision.vit.deit3.dei_t3_small import DeiT3Small

__all__ = [
    "DeiT3Attention",
    "DeiT3Backbone",
    "DeiT3Base",
    "DeiT3Block",
    "DeiT3Huge",
    "DeiT3Large",
    "DeiT3Small",
]
