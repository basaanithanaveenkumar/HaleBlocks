"""``ResNeSt200``: ResNeSt-200: depths=[3,24,36,3], radix=2, ~70M params, embed_dim=2048."""

from __future__ import annotations

from hale_core.nn.vision.cnn.resnest.res_ne_st import ResNeSt
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("resnest200")
class ResNeSt200(ResNeSt):
    """ResNeSt-200: depths=[3,24,36,3], radix=2, ~70M params, embed_dim=2048."""

    arch_name = "resnest200"
    embed_dim = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__(layers=[3, 24, 36, 3], **kwargs)
