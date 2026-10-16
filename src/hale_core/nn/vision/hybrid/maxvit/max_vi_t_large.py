"""``MaxViTLarge``: MaxViT-Large: ~212M params. Strong ImageNet accuracy."""

from __future__ import annotations

from hale_core.nn.vision.hybrid.maxvit.max_vi_t_backbone import MaxViTBackbone
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("maxvit_large")
class MaxViTLarge(MaxViTBackbone):
    """MaxViT-Large: ~212M params. Strong ImageNet accuracy."""

    arch_name = "maxvit_large"
    embed_dim = 1024  # 128 * 8

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("dim", 128)
        kwargs.setdefault("depths", [2, 6, 14, 2])
        super().__init__(**kwargs)
