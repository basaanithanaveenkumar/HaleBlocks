"""``ConvNeXtSmall``: ConvNeXt V2 Small: depths=[3,3,27,3], dims=[96,192,384,768]."""

from __future__ import annotations

from hale_core.nn.vision.cnn.convnext.conv_ne_xt_v2 import ConvNeXtV2
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("convnext_small")
class ConvNeXtSmall(ConvNeXtV2):
    """ConvNeXt V2 Small: depths=[3,3,27,3], dims=[96,192,384,768]."""

    arch_name = "convnext_small"
    embed_dim = 768
    feature_dims = [96, 192, 384, 768]

    def __init__(self, **kwargs) -> None:
        super().__init__(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768], **kwargs)
