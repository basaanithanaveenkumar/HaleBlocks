"""``InternImageLarge``: InternImage-L: 223M params, high accuracy."""

from __future__ import annotations

from hale_core.nn.vision.cnn.internimage.intern_image_backbone import InternImageBackbone
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("internimage_large")
class InternImageLarge(InternImageBackbone):
    """InternImage-L: 223M params, high accuracy."""

    arch_name = "internimage_large"

    def __init__(self, **kwargs):
        super().__init__(stem_channels=160, depths=[5, 5, 22, 5], groups=[10, 20, 40, 80], **kwargs)
