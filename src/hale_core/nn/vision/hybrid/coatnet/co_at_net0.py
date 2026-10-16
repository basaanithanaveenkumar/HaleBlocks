"""``CoAtNet0``: CoAtNet-0: ~25M params. Fastest in the CoAtNet family."""

from __future__ import annotations

from hale_core.nn.vision.hybrid.coatnet.co_at_net_backbone import CoAtNetBackbone
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("coatnet_0")
class CoAtNet0(CoAtNetBackbone):
    """CoAtNet-0: ~25M params. Fastest in the CoAtNet family."""

    arch_name = "coatnet_0"
    embed_dim = 768

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("dims", [64, 96, 192, 384, 768])
        kwargs.setdefault("depths", [2, 3, 5, 2, 2])
        super().__init__(**kwargs)
