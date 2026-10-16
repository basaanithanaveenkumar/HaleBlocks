"""``InceptionV1Backbone``: GoogLeNet / Inception-v1: 22 layers, embed_dim=1024."""

from __future__ import annotations

from hale_core.nn.vision.cnn.inception.goog_le_net import GoogLeNet
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("inception_v1")
class InceptionV1Backbone(GoogLeNet):
    """GoogLeNet / Inception-v1: 22 layers, embed_dim=1024."""

    arch_name = "inception_v1"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
