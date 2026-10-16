"""``EfficientNetB5``: EfficientNet-B5 — ~30M params, embed_dim=2048."""

from __future__ import annotations

from hale_core.nn.vision.cnn.efficientnet_b._helpers import _SCALES, _make_divisible
from hale_core.nn.vision.cnn.efficientnet_b.efficient_net_b import EfficientNetB
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("efficientnet_b5")
class EfficientNetB5(EfficientNetB):
    """EfficientNet-B5 — ~30M params, embed_dim=2048."""

    arch_name = "efficientnet_b5"
    embed_dim = _make_divisible(1280 * 1.6, 8)

    def __init__(self, **kwargs):
        dm, wm, _, dr = _SCALES["b5"]
        kwargs.setdefault("drop_rate", dr)
        super().__init__(
            depth_mult=dm, width_mult=wm, head_chs=_make_divisible(1280 * wm, 8), **kwargs
        )
