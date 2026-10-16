"""``ViTTiny``: ViT-Ti/16 — 5.7M params. Fastest ViT; useful for ablations."""

from __future__ import annotations

from hale_core.nn.vision.registry import register_vision_backbone
from hale_core.nn.vision.vit.vit.vi_t_backbone import ViTBackbone


@register_vision_backbone("vit_tiny")
class ViTTiny(ViTBackbone):
    """ViT-Ti/16 — 5.7M params. Fastest ViT; useful for ablations."""

    arch_name = "vit_tiny"
    feature_dims = [192]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 192)
        kwargs.setdefault("depth", 12)
        kwargs.setdefault("num_heads", 3)
        kwargs.setdefault("patch_size", 16)
        super().__init__(**kwargs)
