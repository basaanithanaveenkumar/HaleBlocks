"""``ResNet152``: ResNet-152: Bottleneck × [3,8,36,3] — deep backbone for high-accuracy tasks."""

from __future__ import annotations

from hale_core.nn.vision.cnn.resnet.bottleneck import Bottleneck
from hale_core.nn.vision.cnn.resnet.res_net import ResNet
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("resnet152")
class ResNet152(ResNet):
    """ResNet-152: Bottleneck × [3,8,36,3] — deep backbone for high-accuracy tasks."""

    arch_name = "resnet152"
    embed_dim = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__(Bottleneck, [3, 8, 36, 3], **kwargs)
