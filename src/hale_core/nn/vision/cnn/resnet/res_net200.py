"""``ResNet200``: ResNet-200: Bottleneck × [3,24,36,3] — very deep, used in scaling studies."""

from __future__ import annotations

from hale_core.nn.vision.cnn.resnet.bottleneck import Bottleneck
from hale_core.nn.vision.cnn.resnet.res_net import ResNet
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("resnet200")
class ResNet200(ResNet):
    """ResNet-200: Bottleneck × [3,24,36,3] — very deep, used in scaling studies."""

    arch_name = "resnet200"
    embed_dim = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__(Bottleneck, [3, 24, 36, 3], **kwargs)
