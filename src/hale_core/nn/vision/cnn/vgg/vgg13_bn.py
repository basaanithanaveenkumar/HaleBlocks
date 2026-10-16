"""``VGG13BN``: VGG13BN (VGGNet Vision Backbones)."""

from __future__ import annotations

from hale_core.nn.vision.cnn.vgg.vgg import VGG
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("vgg13_bn")
class VGG13BN(VGG):
    arch_name = "vgg13_bn"

    def __init__(self, **kwargs):
        super().__init__("vgg13", batch_norm=True, **kwargs)
