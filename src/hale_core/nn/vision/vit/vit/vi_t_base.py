"""``ViTBase``: ViT-B/16 — 86M params. The standard reference ViT."""

from __future__ import annotations

from hale_core.nn.vision.registry import register_vision_backbone
from hale_core.nn.vision.vit.vit.vi_t_backbone import ViTBackbone


@register_vision_backbone("vit_base")
class ViTBase(ViTBackbone):
    """ViT-B/16 — 86M params. The standard reference ViT."""

    arch_name = "vit_base"
    feature_dims = [768]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 768)
        kwargs.setdefault("depth", 12)
        kwargs.setdefault("num_heads", 12)
        kwargs.setdefault("patch_size", 16)
        super().__init__(**kwargs)
