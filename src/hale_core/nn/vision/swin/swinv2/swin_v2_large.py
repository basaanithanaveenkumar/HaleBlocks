"""``SwinV2Large``: SwinV2-Large: 197M params. Strong ImageNet-22K pretrain."""

from __future__ import annotations

from hale_core.nn.vision.registry import register_vision_backbone
from hale_core.nn.vision.swin.swinv2.swin_v2_backbone import SwinV2Backbone


@register_vision_backbone("swinv2_large")
class SwinV2Large(SwinV2Backbone):
    """SwinV2-Large: 197M params. Strong ImageNet-22K pretrain."""

    arch_name = "swinv2_large"
    embed_dim = 1536

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 192)
        kwargs.setdefault("depths", [2, 2, 18, 2])
        kwargs.setdefault("num_heads", [6, 12, 24, 48])
        kwargs.setdefault("window_size", 12)
        super().__init__(**kwargs)
