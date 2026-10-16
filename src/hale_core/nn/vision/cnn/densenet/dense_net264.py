"""``DenseNet264``: DenseNet-264: (6,12,64,48) layers, k=32, embed_dim=2688."""

from __future__ import annotations

from hale_core.nn.vision.cnn.densenet.dense_net import DenseNet
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("densenet264")
class DenseNet264(DenseNet):
    """DenseNet-264: (6,12,64,48) layers, k=32, embed_dim=2688."""

    arch_name = "densenet264"
    embed_dim = 2688
    feature_dims = [256, 512, 2112, 2688]

    def __init__(self, **kwargs):
        super().__init__(block_config=(6, 12, 64, 48), growth_rate=32, init_features=64, **kwargs)
