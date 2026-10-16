"""``ResNet18``: ResNet-18: BasicBlock × [2,2,2,2] — lightweight, fast, embed_dim=512."""

from __future__ import annotations

from hale_core.nn.vision.cnn.resnet.basic_block import BasicBlock
from hale_core.nn.vision.cnn.resnet.res_net import ResNet
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("resnet18")
class ResNet18(ResNet):
    """ResNet-18: BasicBlock × [2,2,2,2] — lightweight, fast, embed_dim=512."""

    arch_name = "resnet18"
    embed_dim = 512
    feature_dims = [64, 128, 256, 512]

    def __init__(self, **kwargs) -> None:
        super().__init__(BasicBlock, [2, 2, 2, 2], **kwargs)
