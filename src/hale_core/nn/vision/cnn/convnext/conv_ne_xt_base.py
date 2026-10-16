"""``ConvNeXtBase``: ConvNeXt V2 Base: depths=[3,3,27,3], dims=[128,256,512,1024]."""

from __future__ import annotations

from hale_core.nn.vision.cnn.convnext.conv_ne_xt_v2 import ConvNeXtV2
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("convnext_base")
class ConvNeXtBase(ConvNeXtV2):
    """ConvNeXt V2 Base: depths=[3,3,27,3], dims=[128,256,512,1024]."""

    arch_name = "convnext_base"
    embed_dim = 1024
    feature_dims = [128, 256, 512, 1024]

    def __init__(self, **kwargs) -> None:
        super().__init__(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
