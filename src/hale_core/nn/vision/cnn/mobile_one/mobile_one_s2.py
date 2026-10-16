"""``MobileOneS2``: MobileOne-S2: higher accuracy, ~2ms."""

from __future__ import annotations

from hale_core.nn.vision.cnn.mobile_one.mobile_one_backbone import MobileOneBackbone
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("mobile_one_s2")
class MobileOneS2(MobileOneBackbone):
    """MobileOne-S2: higher accuracy, ~2ms."""

    arch_name = "mobile_one_s2"

    def __init__(self, **kwargs):
        super().__init__(width_mult=[1.5, 2.0, 2.5, 4.0], num_blocks=1, **kwargs)
