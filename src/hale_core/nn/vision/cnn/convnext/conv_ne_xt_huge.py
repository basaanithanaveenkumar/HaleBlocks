"""``ConvNeXtHuge``: ConvNeXt V2 Huge: depths=[3,3,27,3], dims=[352,704,1408,2816]."""

from __future__ import annotations

from hale_core.nn.vision.cnn.convnext.conv_ne_xt_v2 import ConvNeXtV2
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("convnext_huge")
class ConvNeXtHuge(ConvNeXtV2):
    """ConvNeXt V2 Huge: depths=[3,3,27,3], dims=[352,704,1408,2816]."""

    arch_name = "convnext_huge"
    embed_dim = 2816
    feature_dims = [352, 704, 1408, 2816]

    def __init__(self, **kwargs) -> None:
        super().__init__(depths=[3, 3, 27, 3], dims=[352, 704, 1408, 2816], **kwargs)
