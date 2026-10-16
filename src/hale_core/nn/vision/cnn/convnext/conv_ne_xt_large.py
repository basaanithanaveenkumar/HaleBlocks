"""``ConvNeXtLarge``: ConvNeXt V2 Large: depths=[3,3,27,3], dims=[192,384,768,1536]."""

from __future__ import annotations

from hale_core.nn.vision.cnn.convnext.conv_ne_xt_v2 import ConvNeXtV2
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("convnext_large")
class ConvNeXtLarge(ConvNeXtV2):
    """ConvNeXt V2 Large: depths=[3,3,27,3], dims=[192,384,768,1536]."""

    arch_name = "convnext_large"
    embed_dim = 1536
    feature_dims = [192, 384, 768, 1536]

    def __init__(self, **kwargs) -> None:
        super().__init__(depths=[3, 3, 27, 3], dims=[192, 384, 768, 1536], **kwargs)
