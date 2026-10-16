"""``DenseNet201``: DenseNet-201: (6,12,48,32) layers, k=32, embed_dim=1920."""

from __future__ import annotations

from hale_core.nn.vision.cnn.densenet.dense_net import DenseNet
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("densenet201")
class DenseNet201(DenseNet):
    """DenseNet-201: (6,12,48,32) layers, k=32, embed_dim=1920."""

    arch_name = "densenet201"
    embed_dim = 1920
    feature_dims = [256, 512, 1792, 1920]

    def __init__(self, **kwargs):
        super().__init__(block_config=(6, 12, 48, 32), growth_rate=32, init_features=64, **kwargs)
