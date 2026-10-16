"""``MobileOneS3``: MobileOne-S3: large efficient model."""

from __future__ import annotations

from hale_core.nn.vision.cnn.mobile_one.mobile_one_backbone import MobileOneBackbone
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("mobile_one_s3")
class MobileOneS3(MobileOneBackbone):
    """MobileOne-S3: large efficient model."""

    arch_name = "mobile_one_s3"

    def __init__(self, **kwargs):
        super().__init__(width_mult=[2.0, 2.5, 3.0, 4.0], num_blocks=1, **kwargs)
