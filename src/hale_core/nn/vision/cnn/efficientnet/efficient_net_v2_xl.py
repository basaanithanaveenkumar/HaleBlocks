"""``EfficientNetV2XL``: EfficientNetV2-XL: ~208M params, largest variant with extended training
data.
"""

from __future__ import annotations

from hale_core.nn.vision.cnn.efficientnet._helpers import _xl_stages
from hale_core.nn.vision.cnn.efficientnet.efficient_net_v2_backbone import EfficientNetV2Backbone
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("efficientnet_v2_xl")
class EfficientNetV2XL(EfficientNetV2Backbone):
    """EfficientNetV2-XL: ~208M params, largest variant with extended training data."""

    arch_name = "efficientnet_v2_xl"
    embed_dim = 1280

    def __init__(self, **kwargs):
        super().__init__(stages=_xl_stages(), head_channels=1280, **kwargs)
