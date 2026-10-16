"""``MobileOneS1``: MobileOne-S1: balanced accuracy/speed."""

from __future__ import annotations

from hale_core.nn.vision.cnn.mobile_one.mobile_one_backbone import MobileOneBackbone
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("mobile_one_s1")
class MobileOneS1(MobileOneBackbone):
    """MobileOne-S1: balanced accuracy/speed."""

    arch_name = "mobile_one_s1"

    def __init__(self, **kwargs):
        super().__init__(width_mult=[1.5, 1.5, 2.0, 2.5], num_blocks=1, **kwargs)
