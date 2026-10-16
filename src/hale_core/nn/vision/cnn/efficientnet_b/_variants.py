"""Variants and registrations of :mod:`hale_core.nn.vision.cnn.efficientnet_b` (EfficientNet B0-B7 Vision Backbones)."""

from __future__ import annotations

from hale_core.nn.vision.cnn.efficientnet_b._helpers import _SCALES, _make_divisible
from hale_core.nn.vision.cnn.efficientnet_b.efficient_net_b import EfficientNetB


def _build_b(variant: str, **kwargs):
    dm, wm, _, dr = _SCALES[variant]
    kwargs.setdefault("drop_rate", dr)
    return EfficientNetB(
        depth_mult=dm, width_mult=wm, head_chs=_make_divisible(1280 * wm, 8), **kwargs
    )
