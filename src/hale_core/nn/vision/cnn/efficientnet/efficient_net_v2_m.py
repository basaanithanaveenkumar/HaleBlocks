"""``EfficientNetV2M``: EfficientNetV2-M: ~54M params, balanced accuracy/speed."""

from __future__ import annotations

from hale_core.nn.vision.cnn.efficientnet._helpers import _m_stages
from hale_core.nn.vision.cnn.efficientnet.efficient_net_v2_backbone import EfficientNetV2Backbone
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("efficientnet_v2_m")
class EfficientNetV2M(EfficientNetV2Backbone):
    """EfficientNetV2-M: ~54M params, balanced accuracy/speed."""

    arch_name = "efficientnet_v2_m"
    embed_dim = 1280

    def __init__(self, **kwargs):
        super().__init__(stages=_m_stages(), head_channels=1280, **kwargs)
