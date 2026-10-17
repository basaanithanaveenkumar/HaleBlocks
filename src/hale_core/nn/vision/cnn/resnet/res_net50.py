"""``ResNet50``: ResNet-50: Bottleneck × [3,4,6,3] — standard production backbone,
embed_dim=2048.
"""

from __future__ import annotations

from hale_core.nn.vision.cnn.resnet.bottleneck import Bottleneck
from hale_core.nn.vision.cnn.resnet.res_net import ResNet
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("resnet50")
class ResNet50(ResNet):
    """ResNet-50: Bottleneck × [3,4,6,3] — standard production backbone, embed_dim=2048."""

    arch_name = "resnet50"
    embed_dim = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__(Bottleneck, [3, 4, 6, 3], **kwargs)
