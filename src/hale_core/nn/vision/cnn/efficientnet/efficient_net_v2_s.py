"""``EfficientNetV2S``: EfficientNetV2-S: ~21M params, fast training with Fused-MBConv."""

from __future__ import annotations

from hale_core.nn.vision.cnn.efficientnet._helpers import _s_stages
from hale_core.nn.vision.cnn.efficientnet.efficient_net_v2_backbone import EfficientNetV2Backbone
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("efficientnet_v2_s")
class EfficientNetV2S(EfficientNetV2Backbone):
    """EfficientNetV2-S: ~21M params, fast training with Fused-MBConv."""

    arch_name = "efficientnet_v2_s"
    embed_dim = 1280

    def __init__(self, **kwargs):
        super().__init__(stages=_s_stages(), head_channels=1280, **kwargs)
