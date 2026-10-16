"""``InternImageTiny``: InternImage-T: 30M params, fast."""

from __future__ import annotations

from hale_core.nn.vision.cnn.internimage.intern_image_backbone import InternImageBackbone
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("internimage_tiny")
class InternImageTiny(InternImageBackbone):
    """InternImage-T: 30M params, fast."""

    arch_name = "internimage_tiny"

    def __init__(self, **kwargs):
        super().__init__(stem_channels=64, depths=[4, 4, 18, 4], groups=[4, 8, 16, 32], **kwargs)
