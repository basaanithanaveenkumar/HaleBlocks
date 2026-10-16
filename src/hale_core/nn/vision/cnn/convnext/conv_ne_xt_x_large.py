"""``ConvNeXtXLarge``: ConvNeXt V2 XLarge: depths=[3,3,27,3], dims=[256,512,1024,2048]."""

from __future__ import annotations

from hale_core.nn.vision.cnn.convnext.conv_ne_xt_v2 import ConvNeXtV2
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("convnext_xlarge")
class ConvNeXtXLarge(ConvNeXtV2):
    """ConvNeXt V2 XLarge: depths=[3,3,27,3], dims=[256,512,1024,2048]."""

    arch_name = "convnext_xlarge"
    embed_dim = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], **kwargs)
