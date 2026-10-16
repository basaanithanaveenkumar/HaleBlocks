"""``DINOViTSmall``: DINOv2 ViT-S/14 — 22M params."""

from __future__ import annotations

from hale_core.nn.vision.registry import register_vision_backbone
from hale_core.nn.vision.vit.dino.din_ov2_backbone import DINOv2Backbone

# Reuse PatchEmbed and MLP from sibling vit module (avoid duplication)


@register_vision_backbone("dino_vit_small")
class DINOViTSmall(DINOv2Backbone):
    """DINOv2 ViT-S/14 — 22M params."""

    arch_name = "dino_vit_small"
    feature_dims = [384]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 384)
        kwargs.setdefault("depth", 12)
        kwargs.setdefault("num_heads", 6)
        kwargs.setdefault("patch_size", 14)
        kwargs.setdefault("n_register", 4)
        kwargs.setdefault("ls_init", 1e-5)
        super().__init__(**kwargs)
