"""``InternImageHuge``: InternImage-H: 1.08B params — largest vision backbone in the family."""

from __future__ import annotations

from hale_core.nn.vision.cnn.internimage.intern_image_backbone import InternImageBackbone
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("internimage_huge")
class InternImageHuge(InternImageBackbone):
    """InternImage-H: 1.08B params — largest vision backbone in the family."""

    arch_name = "internimage_huge"

    def __init__(self, **kwargs):
        super().__init__(
            stem_channels=320, depths=[6, 6, 32, 6], groups=[20, 40, 80, 160], **kwargs
        )
