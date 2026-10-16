"""``InternImageBase``: InternImage-B: 97M params."""

from __future__ import annotations

from hale_core.nn.vision.cnn.internimage.intern_image_backbone import InternImageBackbone
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("internimage_base")
class InternImageBase(InternImageBackbone):
    """InternImage-B: 97M params."""

    arch_name = "internimage_base"

    def __init__(self, **kwargs):
        super().__init__(stem_channels=112, depths=[4, 4, 21, 4], groups=[7, 14, 28, 56], **kwargs)
