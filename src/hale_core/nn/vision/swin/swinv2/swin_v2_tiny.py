"""``SwinV2Tiny``: SwinV2-Tiny: 28M params. Fast, suitable for mobile / edge."""

from __future__ import annotations

from hale_core.nn.vision.registry import register_vision_backbone
from hale_core.nn.vision.swin.swinv2.swin_v2_backbone import SwinV2Backbone


@register_vision_backbone("swinv2_tiny")
class SwinV2Tiny(SwinV2Backbone):
    """SwinV2-Tiny: 28M params. Fast, suitable for mobile / edge."""

    arch_name = "swinv2_tiny"
    embed_dim = 768  # 96 * 8

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 96)
        kwargs.setdefault("depths", [2, 2, 6, 2])
        kwargs.setdefault("num_heads", [3, 6, 12, 24])
        kwargs.setdefault("window_size", 8)
        super().__init__(**kwargs)
