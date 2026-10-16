"""``ResNeXt101_32x8d``: ResNeXt-101 (32×8d): Bottleneck×[3,4,23,3], C=32, D=8, embed_dim=2048."""

from __future__ import annotations

from hale_core.nn.vision.cnn.resnext.res_ne_xt import ResNeXt
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("resnext101_32x8d")
class ResNeXt101_32x8d(ResNeXt):
    """ResNeXt-101 (32×8d): Bottleneck×[3,4,23,3], C=32, D=8, embed_dim=2048."""

    arch_name = "resnext101_32x8d"
    embed_dim = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs):
        super().__init__([3, 4, 23, 3], cardinality=32, base_width=8, **kwargs)
