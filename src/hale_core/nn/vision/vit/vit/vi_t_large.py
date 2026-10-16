"""``ViTLarge``: ViT-L/16 — 307M params. High-capacity model for demanding tasks."""

from __future__ import annotations

from hale_core.nn.vision.registry import register_vision_backbone
from hale_core.nn.vision.vit.vit.vi_t_backbone import ViTBackbone


@register_vision_backbone("vit_large")
class ViTLarge(ViTBackbone):
    """ViT-L/16 — 307M params. High-capacity model for demanding tasks."""

    arch_name = "vit_large"
    feature_dims = [1024]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 1024)
        kwargs.setdefault("depth", 24)
        kwargs.setdefault("num_heads", 16)
        kwargs.setdefault("patch_size", 16)
        super().__init__(**kwargs)
