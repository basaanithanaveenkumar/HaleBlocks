"""``MobileNetV1_025``: MobileNetV1 α=0.25 — ultra-lightweight, ~0.5M params, embed_dim=256."""

from __future__ import annotations

from hale_core.nn.vision.cnn.mobilenet_v1.mobile_net_v1 import MobileNetV1
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("mobilenet_v1_025")
class MobileNetV1_025(MobileNetV1):
    """MobileNetV1 α=0.25 — ultra-lightweight, ~0.5M params, embed_dim=256."""

    arch_name = "mobilenet_v1_025"
    embed_dim = 256
    feature_dims = [16, 32, 64, 128, 256]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("alpha", 0.25)
        super().__init__(**kwargs)
