"""``InternImageSmall``: InternImage-S: 50M params."""

from __future__ import annotations

from hale_core.nn.vision.cnn.internimage.intern_image_backbone import InternImageBackbone
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("internimage_small")
class InternImageSmall(InternImageBackbone):
    """InternImage-S: 50M params."""

    arch_name = "internimage_small"

    def __init__(self, **kwargs):
        super().__init__(stem_channels=80, depths=[4, 4, 21, 4], groups=[5, 10, 20, 40], **kwargs)
