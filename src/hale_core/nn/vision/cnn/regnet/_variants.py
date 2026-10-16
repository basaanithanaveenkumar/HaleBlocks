"""Variants and registrations of :mod:`hale_core.nn.vision.cnn.regnet` (RegNet Vision Backbones)."""

from __future__ import annotations

from hale_core.nn.vision.cnn.regnet._helpers import _REGNETX_CFGS, _SUFFIXES
from hale_core.nn.vision.cnn.regnet.reg_net import RegNet
from hale_core.nn.vision.registry import register_vision_backbone


def _make_regnetx(key: str, name: str) -> type:
    cfg = _REGNETX_CFGS[key]
    w = cfg["widths"][-1]

    class _RegNetX(RegNet):
        arch_name = name
        embed_dim = w
        feature_dims = cfg["widths"]

        def __init__(self, **kwargs):
            super().__init__(
                depths=cfg["depths"],
                widths=cfg["widths"],
                group_width=cfg["group_width"],
                use_se=False,
                **kwargs,
            )

    _RegNetX.__name__ = _RegNetX.__qualname__ = name.replace("_", "").upper()
    return _RegNetX


def _make_regnety(key: str, name: str) -> type:
    cfg = _REGNETX_CFGS[key]  # same structural config; only differs in SE
    w = cfg["widths"][-1]

    class _RegNetY(RegNet):
        arch_name = name
        embed_dim = w
        feature_dims = cfg["widths"]

        def __init__(self, **kwargs):
            super().__init__(
                depths=cfg["depths"],
                widths=cfg["widths"],
                group_width=cfg["group_width"],
                use_se=True,
                **kwargs,
            )

    _RegNetY.__name__ = _RegNetY.__qualname__ = name.replace("_", "").upper()
    return _RegNetY


for _sfx in _SUFFIXES:
    _rx_name = f"regnetx_{_sfx}"
    _ry_name = f"regnety_{_sfx}"
    _rx_cls = _make_regnetx(_sfx, _rx_name)
    _ry_cls = _make_regnety(_sfx, _ry_name)
    register_vision_backbone(_rx_name)(_rx_cls)
    register_vision_backbone(_ry_name)(_ry_cls)
    # Also expose in module namespace so imports work
    globals()[_rx_cls.__name__] = _rx_cls
    globals()[_ry_cls.__name__] = _ry_cls
