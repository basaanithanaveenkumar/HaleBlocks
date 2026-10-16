"""``ViTHuge``: ViT-H/14 — 632M params. Maximum standard ViT size."""

from __future__ import annotations

from hale_core.nn.vision.registry import register_vision_backbone
from hale_core.nn.vision.vit.vit.vi_t_backbone import ViTBackbone


@register_vision_backbone("vit_huge")
class ViTHuge(ViTBackbone):
    """ViT-H/14 — 632M params. Maximum standard ViT size."""

    arch_name = "vit_huge"
    feature_dims = [1280]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 1280)
        kwargs.setdefault("depth", 32)
        kwargs.setdefault("num_heads", 16)
        kwargs.setdefault("patch_size", 14)
        super().__init__(**kwargs)
