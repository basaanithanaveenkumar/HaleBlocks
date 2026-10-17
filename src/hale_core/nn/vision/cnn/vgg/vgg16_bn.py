"""``VGG16BN``: VGG-16 with batch normalisation (registered as ``vgg16_bn``)."""

from __future__ import annotations

from hale_core.nn.vision.cnn.vgg.vgg import VGG
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("vgg16_bn")
class VGG16BN(VGG):
    """VGG-16 with batch normalisation (registered as ``vgg16_bn``)."""

    arch_name = "vgg16_bn"

    def __init__(self, **kwargs):
        super().__init__("vgg16", batch_norm=True, **kwargs)
