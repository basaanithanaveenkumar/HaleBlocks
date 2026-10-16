"""``MobileOneS4``: MobileOne-S4: largest, highest accuracy in the family."""

from __future__ import annotations

from hale_core.nn.vision.cnn.mobile_one.mobile_one_backbone import MobileOneBackbone
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("mobile_one_s4")
class MobileOneS4(MobileOneBackbone):
    """MobileOne-S4: largest, highest accuracy in the family."""

    arch_name = "mobile_one_s4"

    def __init__(self, **kwargs):
        super().__init__(width_mult=[3.0, 3.5, 3.5, 4.0], num_blocks=1, **kwargs)
