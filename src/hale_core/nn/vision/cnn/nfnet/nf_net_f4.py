"""``NFNetF4``: NFNet-F4: depths=[5,10,30,15] — 316M params."""

from __future__ import annotations

from hale_core.nn.vision.cnn.nfnet.nf_net import NFNet
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("nfnet_f4")
class NFNetF4(NFNet):
    """NFNet-F4: depths=[5,10,30,15] — 316M params."""

    arch_name = "nfnet_f4"
    embed_dim = 1536
    feature_dims = [256, 512, 1536, 1536]

    def __init__(self, **kwargs) -> None:
        super().__init__(variant="f4", **kwargs)
