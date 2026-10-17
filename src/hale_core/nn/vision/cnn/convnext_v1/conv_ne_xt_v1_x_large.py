"""``ConvNeXtV1XLarge``: ConvNeXt V1 XLarge: depths=[3,3,27,3], dim=256, layer_scale=1e-6 — 350M
params, 87.0%.
"""

from __future__ import annotations

from hale_core.nn.vision.cnn.convnext_v1.conv_ne_xt_v1 import ConvNeXtV1
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("convnext_v1_xlarge")
class ConvNeXtV1XLarge(ConvNeXtV1):
    """ConvNeXt V1 XLarge: depths=[3,3,27,3], dim=256, layer_scale=1e-6 — 350M params, 87.0%."""

    arch_name = "convnext_v1_xlarge"
    embed_dim = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__(depths=[3, 3, 27, 3], dim=256, layer_scale_init_value=1e-6, **kwargs)
