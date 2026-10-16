"""``MaxViTBase``: MaxViT-Base: ~120M params. Higher accuracy with deeper stages."""

from __future__ import annotations

from hale_core.nn.vision.hybrid.maxvit.max_vi_t_backbone import MaxViTBackbone
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("maxvit_base")
class MaxViTBase(MaxViTBackbone):
    """MaxViT-Base: ~120M params. Higher accuracy with deeper stages."""

    arch_name = "maxvit_base"
    embed_dim = 768

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("dim", 96)
        kwargs.setdefault("depths", [2, 6, 14, 2])
        super().__init__(**kwargs)
