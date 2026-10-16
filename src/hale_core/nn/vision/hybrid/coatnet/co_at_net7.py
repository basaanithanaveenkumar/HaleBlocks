"""``CoAtNet7``: CoAtNet-7: ~2.4B params. Giant model matching ViT-22B scale."""

from __future__ import annotations

from hale_core.nn.vision.hybrid.coatnet.co_at_net_backbone import CoAtNetBackbone
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("coatnet_7")
class CoAtNet7(CoAtNetBackbone):
    """CoAtNet-7: ~2.4B params. Giant model matching ViT-22B scale."""

    arch_name = "coatnet_7"
    embed_dim = 1536

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("dims", [192, 192, 384, 768, 1536])
        kwargs.setdefault("depths", [2, 12, 28, 2, 2])
        super().__init__(**kwargs)
