"""``ResNet34``: ResNet-34: BasicBlock × [3,4,6,3] — better accuracy than 18, same embed_dim."""

from __future__ import annotations

from hale_core.nn.vision.cnn.resnet.basic_block import BasicBlock
from hale_core.nn.vision.cnn.resnet.res_net import ResNet
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("resnet34")
class ResNet34(ResNet):
    """ResNet-34: BasicBlock × [3,4,6,3] — better accuracy than 18, same embed_dim."""

    arch_name = "resnet34"
    embed_dim = 512
    feature_dims = [64, 128, 256, 512]

    def __init__(self, **kwargs) -> None:
        super().__init__(BasicBlock, [3, 4, 6, 3], **kwargs)
