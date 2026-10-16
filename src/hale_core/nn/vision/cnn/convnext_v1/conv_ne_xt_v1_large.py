"""``ConvNeXtV1Large``: ConvNeXt V1 Large: depths=[3,3,27,3], dim=192, layer_scale=1e-6 — 198M params, 84.3%."""

from __future__ import annotations

from hale_core.nn.vision.cnn.convnext_v1.conv_ne_xt_v1 import ConvNeXtV1
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("convnext_v1_large")
class ConvNeXtV1Large(ConvNeXtV1):
    """ConvNeXt V1 Large: depths=[3,3,27,3], dim=192, layer_scale=1e-6 — 198M params, 84.3%."""

    arch_name = "convnext_v1_large"
    embed_dim = 1536
    feature_dims = [192, 384, 768, 1536]

    def __init__(self, **kwargs) -> None:
        super().__init__(depths=[3, 3, 27, 3], dim=192, layer_scale_init_value=1e-6, **kwargs)
