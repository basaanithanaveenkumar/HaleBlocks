"""``VGG19BN``: VGG-19 with batch normalisation (registered as ``vgg19_bn``)."""

from __future__ import annotations

from hale_core.nn.vision.cnn.vgg.vgg import VGG
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("vgg19_bn")
class VGG19BN(VGG):
    """VGG-19 with batch normalisation (registered as ``vgg19_bn``)."""

    arch_name = "vgg19_bn"

    def __init__(self, **kwargs):
        super().__init__("vgg19", batch_norm=True, **kwargs)
