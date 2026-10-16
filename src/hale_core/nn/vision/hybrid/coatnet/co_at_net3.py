"""``CoAtNet3``: CoAtNet-3: ~168M params. Large-scale model for high accuracy."""

from __future__ import annotations

from hale_core.nn.vision.hybrid.coatnet.co_at_net_backbone import CoAtNetBackbone
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("coatnet_3")
class CoAtNet3(CoAtNetBackbone):
    """CoAtNet-3: ~168M params. Large-scale model for high accuracy."""

    arch_name = "coatnet_3"
    embed_dim = 1536

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("dims", [192, 192, 384, 768, 1536])
        kwargs.setdefault("depths", [2, 6, 14, 2, 2])
        super().__init__(**kwargs)
