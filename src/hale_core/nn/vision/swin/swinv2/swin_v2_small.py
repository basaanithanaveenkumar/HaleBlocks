"""``SwinV2Small``: SwinV2-Small: 50M params. Good balance of accuracy/speed."""

from __future__ import annotations

from hale_core.nn.vision.registry import register_vision_backbone
from hale_core.nn.vision.swin.swinv2.swin_v2_backbone import SwinV2Backbone


@register_vision_backbone("swinv2_small")
class SwinV2Small(SwinV2Backbone):
    """SwinV2-Small: 50M params. Good balance of accuracy/speed."""

    arch_name = "swinv2_small"
    embed_dim = 768

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 96)
        kwargs.setdefault("depths", [2, 2, 18, 2])
        kwargs.setdefault("num_heads", [3, 6, 12, 24])
        kwargs.setdefault("window_size", 8)
        super().__init__(**kwargs)
