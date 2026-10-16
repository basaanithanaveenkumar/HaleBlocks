"""``InceptionV3Backbone``: Inception-v3: factorised convolutions, embed_dim=2048."""

from __future__ import annotations

from hale_core.nn.vision.cnn.inception.inception_v3 import InceptionV3
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("inception_v3")
class InceptionV3Backbone(InceptionV3):
    """Inception-v3: factorised convolutions, embed_dim=2048."""

    arch_name = "inception_v3"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
