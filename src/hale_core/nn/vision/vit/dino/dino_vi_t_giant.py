"""``DINOViTGiant``: DINOv2 ViT-G/14 — 1.1B params.  The flagship DINOv2 model."""

from __future__ import annotations

from hale_core.nn.vision.registry import register_vision_backbone
from hale_core.nn.vision.vit.dino.din_ov2_backbone import DINOv2Backbone

# Reuse PatchEmbed and MLP from sibling vit module (avoid duplication)


@register_vision_backbone("dino_vit_giant")
class DINOViTGiant(DINOv2Backbone):
    """DINOv2 ViT-G/14 — 1.1B params.  The flagship DINOv2 model."""

    arch_name = "dino_vit_giant"
    feature_dims = [1536]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 1536)
        kwargs.setdefault("depth", 40)
        kwargs.setdefault("num_heads", 24)
        kwargs.setdefault("patch_size", 14)
        kwargs.setdefault("n_register", 4)
        kwargs.setdefault("ls_init", 1e-5)
        super().__init__(**kwargs)
