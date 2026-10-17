"""``VGG19``: VGG-19 (registered as ``vgg19``)."""

from __future__ import annotations

from hale_core.nn.vision.cnn.vgg.vgg import VGG
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("vgg19")
class VGG19(VGG):
    """VGG-19 (registered as ``vgg19``)."""
    arch_name = "vgg19"

    def __init__(self, **kwargs):
        super().__init__("vgg19", batch_norm=False, **kwargs)
