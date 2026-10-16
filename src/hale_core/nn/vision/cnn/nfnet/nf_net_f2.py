"""``NFNetF2``: NFNet-F2: depths=[3,6,18,9] — 194M params."""

from __future__ import annotations

from hale_core.nn.vision.cnn.nfnet.nf_net import NFNet
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("nfnet_f2")
class NFNetF2(NFNet):
    """NFNet-F2: depths=[3,6,18,9] — 194M params."""

    arch_name = "nfnet_f2"
    embed_dim = 1536
    feature_dims = [256, 512, 1536, 1536]

    def __init__(self, **kwargs) -> None:
        super().__init__(variant="f2", **kwargs)
