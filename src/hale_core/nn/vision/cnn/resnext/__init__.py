"""
ResNeXt Vision Backbones
========================

Implements the ResNeXt family (Xie et al. CVPR 2017, "Aggregated Residual
Transformations for Deep Neural Networks").

The key insight: instead of a single "wider" convolution, use a group of
"narrower" convolutions (cardinality C) — aggregated transformations that
are computationally equivalent to a grouped convolution.  This improves
accuracy without increasing FLOPs, adding a new dimension to scaling beyond
depth and width.

Architecture
------------
ResNeXt uses the same 4-stage hierarchy as ResNet but replaces the 3×3 conv
in each Bottleneck with a grouped convolution:

  ResNeXtBottleneck (32×4d, i.e. 32 groups × 4 width):
    1×1 Conv(in → C*D)          — reduce to grouped width
    3×3 GroupConv(C*D → C*D, groups=C) — aggregated transformations
    1×1 Conv(C*D → planes×4)   — expand to output channels
    + residual shortcut

  where C = cardinality (groups), D = base_width per group
  For 32×4d: C=32, D=4, so mid_ch = C*D = 128
  For 32×8d: C=32, D=8, so mid_ch = C*D = 256

Stage structure identical to ResNet:
  Stem → Stage 0 (64→256) → Stage 1 (128→512) → Stage 2 (256→1024) → Stage 3 (512→2048)
  embed_dim = 2048 for all variants (same as ResNet50/101).

Registered names
----------------
  resnext50_32x4d   — [3, 4, 6, 3]  × ResNeXtBottleneck(C=32, D=4)
  resnext101_32x4d  — [3, 4, 23, 3] × ResNeXtBottleneck(C=32, D=4)
  resnext101_32x8d  — [3, 4, 23, 3] × ResNeXtBottleneck(C=32, D=8)

Citation
--------
  Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2017). Aggregated
  Residual Transformations for Deep Neural Networks. CVPR 2017. arXiv:1611.05431.
"""

from hale_core.nn.vision.cnn.resnext.res_ne_xt import ResNeXt
from hale_core.nn.vision.cnn.resnext.res_ne_xt50_32x4d import ResNeXt50_32x4d
from hale_core.nn.vision.cnn.resnext.res_ne_xt101_32x4d import ResNeXt101_32x4d
from hale_core.nn.vision.cnn.resnext.res_ne_xt101_32x8d import ResNeXt101_32x8d
from hale_core.nn.vision.cnn.resnext.res_ne_xt_bottleneck import ResNeXtBottleneck

__all__ = [
    "ResNeXt",
    "ResNeXt101_32x4d",
    "ResNeXt101_32x8d",
    "ResNeXt50_32x4d",
    "ResNeXtBottleneck",
]
