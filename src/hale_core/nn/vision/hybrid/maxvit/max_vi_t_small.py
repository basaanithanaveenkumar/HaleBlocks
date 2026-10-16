"""``MaxViTSmall``: MaxViT-Small: ~69M params. Good accuracy/speed trade-off."""

from __future__ import annotations

from hale_core.nn.vision.hybrid.maxvit.max_vi_t_backbone import MaxViTBackbone
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("maxvit_small")
class MaxViTSmall(MaxViTBackbone):
    """MaxViT-Small: ~69M params. Good accuracy/speed trade-off."""

    arch_name = "maxvit_small"
    embed_dim = 768  # 96 * 8

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("dim", 96)
        kwargs.setdefault("depths", [2, 2, 5, 2])
        super().__init__(**kwargs)
