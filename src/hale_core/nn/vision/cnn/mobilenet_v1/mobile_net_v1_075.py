"""``MobileNetV1_075``: MobileNetV1 α=0.75 — balanced, ~2.6M params, embed_dim=768."""

from __future__ import annotations

from hale_core.nn.vision.cnn.mobilenet_v1.mobile_net_v1 import MobileNetV1
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("mobilenet_v1_075")
class MobileNetV1_075(MobileNetV1):
    """MobileNetV1 α=0.75 — balanced, ~2.6M params, embed_dim=768."""

    arch_name = "mobilenet_v1_075"
    embed_dim = 768
    feature_dims = [48, 96, 192, 384, 768]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("alpha", 0.75)
        super().__init__(**kwargs)
