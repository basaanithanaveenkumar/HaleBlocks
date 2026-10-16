"""``CoAtNet2``: CoAtNet-2: ~75M params. Wider with 128-dim stem."""

from __future__ import annotations

from hale_core.nn.vision.hybrid.coatnet.co_at_net_backbone import CoAtNetBackbone
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("coatnet_2")
class CoAtNet2(CoAtNetBackbone):
    """CoAtNet-2: ~75M params. Wider with 128-dim stem."""

    arch_name = "coatnet_2"
    embed_dim = 1024

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("dims", [128, 128, 256, 512, 1024])
        kwargs.setdefault("depths", [2, 6, 14, 2, 2])
        super().__init__(**kwargs)
