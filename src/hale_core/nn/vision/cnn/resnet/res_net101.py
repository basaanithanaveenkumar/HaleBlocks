"""``ResNet101``: ResNet-101: Bottleneck × [3,4,23,3] — stronger features, 23 blocks in stage2."""

from __future__ import annotations

from hale_core.nn.vision.cnn.resnet.bottleneck import Bottleneck
from hale_core.nn.vision.cnn.resnet.res_net import ResNet
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("resnet101")
class ResNet101(ResNet):
    """ResNet-101: Bottleneck × [3,4,23,3] — stronger features, 23 blocks in stage2."""

    arch_name = "resnet101"
    embed_dim = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__(Bottleneck, [3, 4, 23, 3], **kwargs)
