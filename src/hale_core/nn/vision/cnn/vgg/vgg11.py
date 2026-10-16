"""``VGG11``: VGG11 (VGGNet Vision Backbones)."""

from __future__ import annotations

from hale_core.nn.vision.cnn.vgg.vgg import VGG
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("vgg11")
class VGG11(VGG):
    arch_name = "vgg11"

    def __init__(self, **kwargs):
        super().__init__("vgg11", batch_norm=False, **kwargs)
