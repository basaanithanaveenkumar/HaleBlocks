"""Shared helpers of :mod:`hale_core.nn.vision.cnn.efficientnet` (EfficientNetV2 Vision Backbones (Tan & Le 2021))."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class _StageConfig:
    """Configuration for one EfficientNetV2 stage."""

    block_type: str  # "fused" or "mbconv"
    in_chs: int
    out_chs: int
    stride: int
    expand_ratio: int
    kernel_size: int
    num_blocks: int
    se_ratio: float  # 0 for fused, 0.25 for mbconv


def _make_divisible(v: float, divisor: int = 8) -> int:
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


def _s_stages():
    return [
        _StageConfig("fused", 24, 24, 1, 1, 3, 2, 0.0),
        _StageConfig("fused", 24, 48, 2, 4, 3, 4, 0.0),
        _StageConfig("fused", 48, 64, 2, 4, 3, 4, 0.0),
        _StageConfig("mbconv", 64, 128, 2, 4, 3, 6, 0.25),
        _StageConfig("mbconv", 128, 160, 1, 6, 3, 9, 0.25),
        _StageConfig("mbconv", 160, 256, 2, 6, 3, 15, 0.25),
    ]


def _m_stages():
    return [
        _StageConfig("fused", 24, 24, 1, 1, 3, 3, 0.0),
        _StageConfig("fused", 24, 48, 2, 4, 3, 5, 0.0),
        _StageConfig("fused", 48, 80, 2, 4, 3, 5, 0.0),
        _StageConfig("mbconv", 80, 160, 2, 4, 3, 7, 0.25),
        _StageConfig("mbconv", 160, 176, 1, 6, 3, 14, 0.25),
        _StageConfig("mbconv", 176, 304, 2, 6, 3, 18, 0.25),
        _StageConfig("mbconv", 304, 512, 1, 6, 3, 5, 0.25),
    ]


def _l_stages():
    return [
        _StageConfig("fused", 32, 32, 1, 1, 3, 4, 0.0),
        _StageConfig("fused", 32, 64, 2, 4, 3, 7, 0.0),
        _StageConfig("fused", 64, 96, 2, 4, 3, 7, 0.0),
        _StageConfig("mbconv", 96, 192, 2, 4, 3, 10, 0.25),
        _StageConfig("mbconv", 192, 224, 1, 6, 3, 19, 0.25),
        _StageConfig("mbconv", 224, 384, 2, 6, 3, 25, 0.25),
        _StageConfig("mbconv", 384, 640, 1, 6, 3, 7, 0.25),
    ]


def _xl_stages():
    return [
        _StageConfig("fused", 32, 32, 1, 1, 3, 4, 0.0),
        _StageConfig("fused", 32, 64, 2, 4, 3, 8, 0.0),
        _StageConfig("fused", 64, 96, 2, 4, 3, 8, 0.0),
        _StageConfig("mbconv", 96, 192, 2, 4, 3, 16, 0.25),
        _StageConfig("mbconv", 192, 256, 1, 6, 3, 24, 0.25),
        _StageConfig("mbconv", 256, 512, 2, 6, 3, 32, 0.25),
        _StageConfig("mbconv", 512, 640, 1, 6, 3, 8, 0.25),
    ]
