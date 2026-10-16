"""``ResNeSt50``: ResNeSt-50: depths=[3,4,6,3], radix=2, ~27M params, embed_dim=2048."""

from __future__ import annotations

from hale_core.nn.vision.cnn.resnest.res_ne_st import ResNeSt
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("resnest50")
class ResNeSt50(ResNeSt):
    """ResNeSt-50: depths=[3,4,6,3], radix=2, ~27M params, embed_dim=2048."""

    arch_name = "resnest50"
    embed_dim = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__(layers=[3, 4, 6, 3], **kwargs)
