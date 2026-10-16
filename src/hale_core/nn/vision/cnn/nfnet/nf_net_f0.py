"""``NFNetF0``: NFNet-F0: depths=[1,2,6,3] — smallest, 71M params."""

from __future__ import annotations

from hale_core.nn.vision.cnn.nfnet.nf_net import NFNet
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("nfnet_f0")
class NFNetF0(NFNet):
    """NFNet-F0: depths=[1,2,6,3] — smallest, 71M params."""

    arch_name = "nfnet_f0"
    embed_dim = 1536
    feature_dims = [256, 512, 1536, 1536]

    def __init__(self, **kwargs) -> None:
        super().__init__(variant="f0", **kwargs)
