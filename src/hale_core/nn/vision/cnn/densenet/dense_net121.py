"""``DenseNet121``: DenseNet-121: (6,12,24,16) layers, k=32, embed_dim=1024."""

from __future__ import annotations

from hale_core.nn.vision.cnn.densenet.dense_net import DenseNet
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("densenet121")
class DenseNet121(DenseNet):
    """DenseNet-121: (6,12,24,16) layers, k=32, embed_dim=1024."""

    arch_name = "densenet121"
    embed_dim = 1024
    feature_dims = [256, 512, 1024, 1024]

    def __init__(self, **kwargs):
        super().__init__(block_config=(6, 12, 24, 16), growth_rate=32, init_features=64, **kwargs)
