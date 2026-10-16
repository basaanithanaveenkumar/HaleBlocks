"""``ViTSmall``: ViT-S/16 — 22M params. Good accuracy/speed trade-off."""

from __future__ import annotations

from hale_core.nn.vision.registry import register_vision_backbone
from hale_core.nn.vision.vit.vit.vi_t_backbone import ViTBackbone


@register_vision_backbone("vit_small")
class ViTSmall(ViTBackbone):
    """ViT-S/16 — 22M params. Good accuracy/speed trade-off."""

    arch_name = "vit_small"
    feature_dims = [384]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 384)
        kwargs.setdefault("depth", 12)
        kwargs.setdefault("num_heads", 6)
        kwargs.setdefault("patch_size", 16)
        super().__init__(**kwargs)
