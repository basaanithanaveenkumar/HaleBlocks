"""``NFNetF3``: NFNet-F3: depths=[4,8,24,12] — 255M params."""

from __future__ import annotations

from hale_core.nn.vision.cnn.nfnet.nf_net import NFNet
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("nfnet_f3")
class NFNetF3(NFNet):
    """NFNet-F3: depths=[4,8,24,12] — 255M params."""

    arch_name = "nfnet_f3"
    embed_dim = 1536
    feature_dims = [256, 512, 1536, 1536]

    def __init__(self, **kwargs) -> None:
        super().__init__(variant="f3", **kwargs)
