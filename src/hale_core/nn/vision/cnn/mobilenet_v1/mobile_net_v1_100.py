"""``MobileNetV1_100``: MobileNetV1 α=1.00 — full-width, ~4.2M params, embed_dim=1024."""

from __future__ import annotations

from hale_core.nn.vision.cnn.mobilenet_v1.mobile_net_v1 import MobileNetV1
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("mobilenet_v1_100")
class MobileNetV1_100(MobileNetV1):
    """MobileNetV1 α=1.00 — full-width, ~4.2M params, embed_dim=1024."""

    arch_name = "mobilenet_v1_100"
    embed_dim = 1024
    feature_dims = [64, 128, 256, 512, 1024]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("alpha", 1.0)
        super().__init__(**kwargs)
