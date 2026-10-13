"""
hale_core.nn.vision.vit — ViT-family vision backbone implementations.

Modules
-------
  vit    — Standard ViT (Dosovitskiy 2020): vit_tiny/small/base/large/huge/22b
  dino   — DINOv2 ViT (Oquab 2023): dino_vit_small/base/large/giant
           Key: register tokens, LayerScale, Flash Attention
  eva    — EVA-02 ViT (Fang 2023): eva_02_small/base/large
           Key: 2D RoPE, SwiGLU FFN, Sub-LN on Q/K
  deit3  — DeiT III ViT (Touvron 2022): deit3_small/base/large/huge
           Key: LayerScale, mean-pool (no CLS token)

All modules register their variants on import via @register_vision_backbone.
The registry's _ensure_backbones_imported() calls this package to trigger
registration; direct imports also work.
"""

from hale_core.nn.vision.vit import vit    # noqa: F401 — registers vit_tiny/small/base/large/huge/22b
from hale_core.nn.vision.vit import dino   # noqa: F401 — registers dino_vit_small/base/large/giant
from hale_core.nn.vision.vit import eva    # noqa: F401 — registers eva_02_small/base/large
from hale_core.nn.vision.vit import deit3  # noqa: F401 — registers deit3_small/base/large/huge

__all__ = ["vit", "dino", "eva", "deit3"]
