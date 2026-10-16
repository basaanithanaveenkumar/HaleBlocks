"""``MobileNetV1_050``: MobileNetV1 α=0.50 — lightweight, ~1.3M params, embed_dim=512."""

from __future__ import annotations

from hale_core.nn.vision.cnn.mobilenet_v1.mobile_net_v1 import MobileNetV1
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("mobilenet_v1_050")
class MobileNetV1_050(MobileNetV1):
    """MobileNetV1 α=0.50 — lightweight, ~1.3M params, embed_dim=512."""

    arch_name = "mobilenet_v1_050"
    embed_dim = 512
    feature_dims = [32, 64, 128, 256, 512]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("alpha", 0.50)
        super().__init__(**kwargs)
