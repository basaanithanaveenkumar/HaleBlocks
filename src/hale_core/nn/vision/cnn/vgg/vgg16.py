"""``VGG16``: VGG16 (VGGNet Vision Backbones)."""

from __future__ import annotations

from hale_core.nn.vision.cnn.vgg.vgg import VGG
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("vgg16")
class VGG16(VGG):
    arch_name = "vgg16"

    def __init__(self, **kwargs):
        super().__init__("vgg16", batch_norm=False, **kwargs)
