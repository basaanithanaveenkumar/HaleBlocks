"""``AlexNetBackbone``: AlexNet: 5-stage CNN, embed_dim=4096."""

from __future__ import annotations

from hale_core.nn.vision.cnn.alexnet.alex_net import AlexNet
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("alexnet")
class AlexNetBackbone(AlexNet):
    """AlexNet: 5-stage CNN, embed_dim=4096."""

    arch_name = "alexnet"
    embed_dim = 4096
    feature_dims = [64, 192, 384, 256, 256]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
