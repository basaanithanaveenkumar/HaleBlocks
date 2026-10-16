"""``SEResNet50``: SE-ResNet-50: SEBottleneck × [3,4,6,3], ~28M params, embed_dim=2048."""

from __future__ import annotations

from hale_core.nn.vision.cnn.senet.se_res_net import SEResNet
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("se_resnet50")
class SEResNet50(SEResNet):
    """SE-ResNet-50: SEBottleneck × [3,4,6,3], ~28M params, embed_dim=2048."""

    arch_name = "se_resnet50"
    embed_dim = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__(layers=[3, 4, 6, 3], **kwargs)
