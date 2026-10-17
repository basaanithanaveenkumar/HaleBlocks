"""Shared helpers of :mod:`hale_core.nn.vision.cnn.efficientnet_b` (EfficientNet B0-B7 Vision
Backbones).
"""

from __future__ import annotations

from dataclasses import dataclass


def _make_divisible(v: float, divisor: int = 8) -> int:
    """Round v up to the nearest multiple of divisor (min=divisor)."""
    return max(divisor, int(v + divisor / 2) // divisor * divisor)


@dataclass
class _B0StageConfig:
    """B0 baseline specification for one stage."""

    expand_ratio: int
    kernel_size: int
    stride: int
    out_chs: int  # baseline (B0) output channels
    num_blocks: int  # baseline (B0) repeat count


# B0 stage definitions from Table 1 of the EfficientNet paper
_B0_STAGES: list[_B0StageConfig] = [
    _B0StageConfig(1, 3, 1, 16, 1),  # Stage 1: MBConv1
    _B0StageConfig(6, 3, 2, 24, 2),  # Stage 2: MBConv6
    _B0StageConfig(6, 5, 2, 40, 2),  # Stage 3: MBConv6 k=5
    _B0StageConfig(6, 3, 2, 80, 3),  # Stage 4: MBConv6
    _B0StageConfig(6, 5, 1, 112, 3),  # Stage 5: MBConv6 k=5
    _B0StageConfig(6, 5, 2, 192, 4),  # Stage 6: MBConv6 k=5
    _B0StageConfig(6, 3, 1, 320, 1),  # Stage 7: MBConv6
]


# (depth_mult, width_mult, default_res, dropout)
_SCALES = {
    "b0": (1.0, 1.0, 224, 0.2),
    "b1": (1.1, 1.0, 240, 0.2),
    "b2": (1.2, 1.1, 260, 0.3),
    "b3": (1.4, 1.2, 300, 0.3),
    "b4": (1.8, 1.4, 380, 0.4),
    "b5": (2.2, 1.6, 456, 0.4),
    "b6": (2.6, 1.8, 528, 0.5),
    "b7": (3.1, 2.0, 600, 0.5),
}
