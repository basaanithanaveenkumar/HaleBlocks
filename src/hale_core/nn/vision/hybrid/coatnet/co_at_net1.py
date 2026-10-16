"""``CoAtNet1``: CoAtNet-1: ~42M params. More blocks for better feature quality."""

from __future__ import annotations

from hale_core.nn.vision.hybrid.coatnet.co_at_net_backbone import CoAtNetBackbone
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("coatnet_1")
class CoAtNet1(CoAtNetBackbone):
    """CoAtNet-1: ~42M params. More blocks for better feature quality."""

    arch_name = "coatnet_1"
    embed_dim = 768

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("dims", [64, 96, 192, 384, 768])
        kwargs.setdefault("depths", [2, 6, 14, 2, 2])
        super().__init__(**kwargs)
