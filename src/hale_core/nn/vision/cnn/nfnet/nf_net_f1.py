"""``NFNetF1``: NFNet-F1: depths=[2,4,12,6] — 132M params."""

from __future__ import annotations

from hale_core.nn.vision.cnn.nfnet.nf_net import NFNet
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("nfnet_f1")
class NFNetF1(NFNet):
    """NFNet-F1: depths=[2,4,12,6] — 132M params."""

    arch_name = "nfnet_f1"
    embed_dim = 1536
    feature_dims = [256, 512, 1536, 1536]

    def __init__(self, **kwargs) -> None:
        super().__init__(variant="f1", **kwargs)
