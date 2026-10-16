"""``ConvNeXtV1Base``: ConvNeXt V1 Base: depths=[3,3,27,3], dim=128 — 89M params, 83.8% top-1."""

from __future__ import annotations

from hale_core.nn.vision.cnn.convnext_v1.conv_ne_xt_v1 import ConvNeXtV1
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("convnext_v1_base")
class ConvNeXtV1Base(ConvNeXtV1):
    """ConvNeXt V1 Base: depths=[3,3,27,3], dim=128 — 89M params, 83.8% top-1."""

    arch_name = "convnext_v1_base"
    embed_dim = 1024
    feature_dims = [128, 256, 512, 1024]

    def __init__(self, **kwargs) -> None:
        super().__init__(depths=[3, 3, 27, 3], dim=128, layer_scale_init_value=0.0, **kwargs)
