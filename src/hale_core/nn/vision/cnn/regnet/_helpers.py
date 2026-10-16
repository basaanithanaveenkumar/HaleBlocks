"""Shared helpers of :mod:`hale_core.nn.vision.cnn.regnet` (RegNet Vision Backbones)."""

from __future__ import annotations

# From RegNet paper Table 14 (RegNetX) — (depths, widths, group_width)
_REGNETX_CFGS: dict[str, dict] = {
    "002": dict(depths=[1, 1, 4, 7], widths=[24, 56, 152, 368], group_width=8),
    "004": dict(depths=[1, 2, 7, 12], widths=[32, 64, 160, 384], group_width=16),
    "008": dict(depths=[1, 3, 8, 2], widths=[64, 128, 288, 672], group_width=16),
    "016": dict(depths=[2, 4, 10, 2], widths=[72, 168, 408, 912], group_width=24),
    "032": dict(depths=[2, 6, 15, 2], widths=[96, 192, 432, 1008], group_width=48),
    "064": dict(depths=[2, 4, 10, 1], widths=[168, 392, 784, 1624], group_width=56),
    "080": dict(depths=[2, 5, 14, 2], widths=[80, 240, 720, 1920], group_width=120),
    "120": dict(depths=[2, 5, 11, 1], widths=[224, 448, 896, 2240], group_width=112),
    "160": dict(depths=[2, 6, 13, 1], widths=[256, 512, 896, 2048], group_width=128),
    "320": dict(depths=[2, 7, 13, 1], widths=[336, 672, 1344, 2520], group_width=168),
}


# Programmatically register all 20 variants
_SUFFIXES = ["002", "004", "008", "016", "032", "064", "080", "120", "160", "320"]
