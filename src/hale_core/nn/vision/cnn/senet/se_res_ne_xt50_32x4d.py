"""``SEResNeXt50_32x4d``: SE-ResNeXt-50 (32×4d): cardinality=32, base_width=4, ~27M params."""

from __future__ import annotations

from hale_core.nn.vision.cnn.senet.se_res_net import SEResNet
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("seresnext50_32x4d")
class SEResNeXt50_32x4d(SEResNet):
    """SE-ResNeXt-50 (32×4d): cardinality=32, base_width=4, ~27M params."""

    arch_name = "seresnext50_32x4d"
    embed_dim = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__(layers=[3, 4, 6, 3], groups=32, base_width=4, **kwargs)
