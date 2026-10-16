"""``EfficientNetV2L``: EfficientNetV2-L: ~119M params, high accuracy."""

from __future__ import annotations

from hale_core.nn.vision.cnn.efficientnet._helpers import _l_stages
from hale_core.nn.vision.cnn.efficientnet.efficient_net_v2_backbone import EfficientNetV2Backbone
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("efficientnet_v2_l")
class EfficientNetV2L(EfficientNetV2Backbone):
    """EfficientNetV2-L: ~119M params, high accuracy."""

    arch_name = "efficientnet_v2_l"
    embed_dim = 1280

    def __init__(self, **kwargs):
        super().__init__(stages=_l_stages(), head_channels=1280, **kwargs)
