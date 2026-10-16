"""``SEResNet101``: SE-ResNet-101: SEBottleneck × [3,4,23,3], ~49M params, embed_dim=2048."""

from __future__ import annotations

from hale_core.nn.vision.cnn.senet.se_res_net import SEResNet
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("se_resnet101")
class SEResNet101(SEResNet):
    """SE-ResNet-101: SEBottleneck × [3,4,23,3], ~49M params, embed_dim=2048."""

    arch_name = "se_resnet101"
    embed_dim = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__(layers=[3, 4, 23, 3], **kwargs)
