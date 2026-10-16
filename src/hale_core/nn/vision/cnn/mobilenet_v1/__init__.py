"""
MobileNetV1 Vision Backbones
=============================

Implements the MobileNetV1 family (Howard et al. 2017, "MobileNets: Efficient
Convolutional Neural Networks for Mobile Vision Applications").

Reference: https://arxiv.org/abs/1704.04861

Architecture
------------
MobileNetV1 replaces standard 3×3 convolutions with **Depthwise Separable
Convolutions**, which factorise a spatial + channel operation into two cheaper
operations:

  1. Depthwise conv: k×k, groups=C_in — spatial feature extraction per channel.
  2. Pointwise conv: 1×1 standard     — linear cross-channel combination.

Computation cost ratio vs standard conv:
    DW+PW / Standard = (k²·C_in + C_in·C_out) / (k²·C_in·C_out)
                     = 1/C_out + 1/k²
    For k=3, C_out=512 → ~8.6× reduction.

Stage structure (α=1.0):
  Stem    : Conv(3→32, 3×3, s2) + BN + ReLU → DWSConv(32→64, s1)  → H/2
  Stage 0 : DWSConv(64→128,  s2) → DWSConv(128→128,  s1)           → H/4
  Stage 1 : DWSConv(128→256, s2) → DWSConv(256→256,  s1)           → H/8
  Stage 2 : DWSConv(256→512, s2) → 5×DWSConv(512→512, s1)          → H/16
  Stage 3 : DWSConv(512→1024,s2) → DWSConv(1024→1024,s1)           → H/32
  Head    : GlobalAvgPool → [B, embed_dim]

All channels are scaled by width multiplier α ∈ {0.25, 0.5, 0.75, 1.0}.
  embed_dim = int(1024 * α)
  feature_dims = [int(c*α) for c in [64, 128, 256, 512, 1024]]

Registered names
----------------
  mobilenet_v1_025 — α=0.25, embed_dim=256
  mobilenet_v1_050 — α=0.50, embed_dim=512
  mobilenet_v1_075 — α=0.75, embed_dim=768
  mobilenet_v1_100 — α=1.00, embed_dim=1024

Citation
--------
  Howard, A. G. et al. (2017). MobileNets: Efficient Convolutional Neural
  Networks for Mobile Vision Applications. arXiv:1704.04861.
"""

import hale_core.nn.vision.cnn.mobilenet_v1._helpers  # noqa: F401 - module-level registrations
from hale_core.nn.vision.cnn.mobilenet_v1.mobile_net_v1 import MobileNetV1
from hale_core.nn.vision.cnn.mobilenet_v1.mobile_net_v1_025 import MobileNetV1_025
from hale_core.nn.vision.cnn.mobilenet_v1.mobile_net_v1_050 import MobileNetV1_050
from hale_core.nn.vision.cnn.mobilenet_v1.mobile_net_v1_075 import MobileNetV1_075
from hale_core.nn.vision.cnn.mobilenet_v1.mobile_net_v1_100 import MobileNetV1_100

__all__ = [
    "MobileNetV1",
    "MobileNetV1_025",
    "MobileNetV1_050",
    "MobileNetV1_075",
    "MobileNetV1_100",
]
