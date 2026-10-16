"""``EfficientNetB1``: EfficientNet-B1 — depth scaled, ~7.8M params, embed_dim=1280."""

from __future__ import annotations

from hale_core.nn.vision.cnn.efficientnet_b._helpers import _SCALES
from hale_core.nn.vision.cnn.efficientnet_b.efficient_net_b import EfficientNetB
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("efficientnet_b1")
class EfficientNetB1(EfficientNetB):
    """EfficientNet-B1 — depth scaled, ~7.8M params, embed_dim=1280."""

    arch_name = "efficientnet_b1"
    embed_dim = 1280

    def __init__(self, **kwargs):
        dm, wm, _, dr = _SCALES["b1"]
        kwargs.setdefault("drop_rate", dr)
        super().__init__(depth_mult=dm, width_mult=wm, head_chs=1280, **kwargs)
