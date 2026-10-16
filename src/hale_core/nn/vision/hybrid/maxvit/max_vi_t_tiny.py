"""``MaxViTTiny``: MaxViT-Tiny: ~31M params. Fastest MaxViT variant."""

from __future__ import annotations

from hale_core.nn.vision.hybrid.maxvit.max_vi_t_backbone import MaxViTBackbone
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("maxvit_tiny")
class MaxViTTiny(MaxViTBackbone):
    """MaxViT-Tiny: ~31M params. Fastest MaxViT variant."""

    arch_name = "maxvit_tiny"
    embed_dim = 512  # 64 * 8

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("dim", 64)
        kwargs.setdefault("depths", [2, 2, 5, 2])
        super().__init__(**kwargs)
