"""``NFNetF5``: NFNet-F5: depths=[6,12,36,18] — largest, 377M params, 89.2% top-1."""

from __future__ import annotations

from hale_core.nn.vision.cnn.nfnet.nf_net import NFNet
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("nfnet_f5")
class NFNetF5(NFNet):
    """NFNet-F5: depths=[6,12,36,18] — largest, 377M params, 89.2% top-1."""

    arch_name = "nfnet_f5"
    embed_dim = 1536
    feature_dims = [256, 512, 1536, 1536]

    def __init__(self, **kwargs) -> None:
        super().__init__(variant="f5", **kwargs)
