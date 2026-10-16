"""``MobileOneS0``: MobileOne-S0: smallest variant, 1.6ms on iPhone 12."""

from __future__ import annotations

from hale_core.nn.vision.cnn.mobile_one.mobile_one_backbone import MobileOneBackbone
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("mobile_one_s0")
class MobileOneS0(MobileOneBackbone):
    """MobileOne-S0: smallest variant, 1.6ms on iPhone 12."""

    arch_name = "mobile_one_s0"

    def __init__(self, **kwargs):
        super().__init__(width_mult=[0.75, 1.0, 1.0, 2.0], num_blocks=1, **kwargs)
