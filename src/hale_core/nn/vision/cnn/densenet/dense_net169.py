"""``DenseNet169``: DenseNet-169: (6,12,32,32) layers, k=32, embed_dim=1664."""

from __future__ import annotations

from hale_core.nn.vision.cnn.densenet.dense_net import DenseNet
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("densenet169")
class DenseNet169(DenseNet):
    """DenseNet-169: (6,12,32,32) layers, k=32, embed_dim=1664."""

    arch_name = "densenet169"
    embed_dim = 1664
    feature_dims = [256, 512, 1280, 1664]

    def __init__(self, **kwargs):
        super().__init__(block_config=(6, 12, 32, 32), growth_rate=32, init_features=64, **kwargs)
