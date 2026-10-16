"""``SwinV2Giant``: SwinV2-Giant: 3B params. Highest accuracy in the SwinV2 family."""

from __future__ import annotations

from hale_core.nn.vision.registry import register_vision_backbone
from hale_core.nn.vision.swin.swinv2.swin_v2_backbone import SwinV2Backbone


@register_vision_backbone("swinv2_giant")
class SwinV2Giant(SwinV2Backbone):
    """SwinV2-Giant: 3B params. Highest accuracy in the SwinV2 family."""

    arch_name = "swinv2_giant"
    embed_dim = 1536

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 192)
        kwargs.setdefault("depths", [2, 2, 42, 4])
        kwargs.setdefault("num_heads", [6, 12, 24, 48])
        kwargs.setdefault("window_size", 12)
        super().__init__(**kwargs)
