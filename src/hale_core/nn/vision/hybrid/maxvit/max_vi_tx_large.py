"""``MaxViTXLarge``: MaxViT-XLarge: ~475M params. Largest MaxViT for high-res tasks."""

from __future__ import annotations

from hale_core.nn.vision.hybrid.maxvit.max_vi_t_backbone import MaxViTBackbone
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("maxvit_xlarge")
class MaxViTXLarge(MaxViTBackbone):
    """MaxViT-XLarge: ~475M params. Largest MaxViT for high-res tasks."""

    arch_name = "maxvit_xlarge"
    embed_dim = 1536  # 192 * 8

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("dim", 192)
        kwargs.setdefault("depths", [2, 6, 14, 2])
        super().__init__(**kwargs)
