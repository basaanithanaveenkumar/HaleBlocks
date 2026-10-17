"""``VGG11BN``: VGG-11 with batch normalisation (registered as ``vgg11_bn``)."""

from __future__ import annotations

from hale_core.nn.vision.cnn.vgg.vgg import VGG
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("vgg11_bn")
class VGG11BN(VGG):
    """VGG-11 with batch normalisation (registered as ``vgg11_bn``)."""
    arch_name = "vgg11_bn"

    def __init__(self, **kwargs):
        super().__init__("vgg11", batch_norm=True, **kwargs)
