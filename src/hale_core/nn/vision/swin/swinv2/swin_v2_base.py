"""``SwinV2Base``: SwinV2-Base: 88M params. High accuracy, used in many benchmarks."""

from __future__ import annotations

from hale_core.nn.vision.registry import register_vision_backbone
from hale_core.nn.vision.swin.swinv2.swin_v2_backbone import SwinV2Backbone


@register_vision_backbone("swinv2_base")
class SwinV2Base(SwinV2Backbone):
    """SwinV2-Base: 88M params. High accuracy, used in many benchmarks."""

    arch_name = "swinv2_base"
    embed_dim = 1024

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 128)
        kwargs.setdefault("depths", [2, 2, 18, 2])
        kwargs.setdefault("num_heads", [4, 8, 16, 32])
        kwargs.setdefault("window_size", 8)
        super().__init__(**kwargs)
