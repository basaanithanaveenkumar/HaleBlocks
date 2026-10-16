"""``VGG11BN``: VGG11BN (VGGNet Vision Backbones)."""

from __future__ import annotations

from hale_core.nn.vision.cnn.vgg.vgg import VGG
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("vgg11_bn")
class VGG11BN(VGG):
    arch_name = "vgg11_bn"

    def __init__(self, **kwargs):
        super().__init__("vgg11", batch_norm=True, **kwargs)
