"""``ConvNeXtV1Tiny``: ConvNeXt V1 Tiny: depths=[3,3,9,3], dim=96 — 28M params, 82.1% top-1."""

from __future__ import annotations

from hale_core.nn.vision.cnn.convnext_v1.conv_ne_xt_v1 import ConvNeXtV1
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("convnext_v1_tiny")
class ConvNeXtV1Tiny(ConvNeXtV1):
    """ConvNeXt V1 Tiny: depths=[3,3,9,3], dim=96 — 28M params, 82.1% top-1."""

    arch_name = "convnext_v1_tiny"
    embed_dim = 768
    feature_dims = [96, 192, 384, 768]

    def __init__(self, **kwargs) -> None:
        super().__init__(depths=[3, 3, 9, 3], dim=96, layer_scale_init_value=0.0, **kwargs)
