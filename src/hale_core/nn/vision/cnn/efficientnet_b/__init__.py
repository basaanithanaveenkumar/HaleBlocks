"""
EfficientNet B0-B7 Vision Backbones
======================================

Implements the original EfficientNet family (Tan & Le 2019, "EfficientNet:
Rethinking Model Scaling for Convolutional Neural Networks", ICML 2019).

Reference: https://arxiv.org/abs/1905.11946

Architecture
------------
EfficientNet introduces **compound scaling**: width, depth and input resolution
are scaled simultaneously using a single compound coefficient φ:

    depth    d = α^φ
    width    w = β^φ
    resolution r = γ^φ

Subject to: α × β² × γ² ≈ 2, α≥1, β≥1, γ≥1.
B0 baseline found via NAS; B1-B7 apply compound scaling.

Building block: **MBConv** (Mobile Inverted Bottleneck Convolution) with SE:
  input → expand 1×1 (ratio r) → BN → SiLU
        → depthwise k×k        → BN → SiLU
        → SE(reduction=4)
        → project 1×1          → BN
        → (+residual if stride=1 and in_chs==out_chs)

B0 baseline stage layout (Table 1 of paper):
  Stage 1 : MBConv1, k=3, s=1, out=16,  repeat=1
  Stage 2 : MBConv6, k=3, s=2, out=24,  repeat=2
  Stage 3 : MBConv6, k=5, s=2, out=40,  repeat=2
  Stage 4 : MBConv6, k=3, s=2, out=80,  repeat=3
  Stage 5 : MBConv6, k=5, s=1, out=112, repeat=3
  Stage 6 : MBConv6, k=5, s=2, out=192, repeat=4
  Stage 7 : MBConv6, k=3, s=1, out=320, repeat=1
  Head    : Conv 1×1 → 1280 → GlobalAvgPool → dropout → embedding

Multi-scale feature map outputs:
  feature_maps[0]: stride-4  (after stem)
  feature_maps[1]: stride-8  (after stage 2)
  feature_maps[2]: stride-16 (after stage 4)
  feature_maps[3]: stride-32 (after stage 6+7+head)

Compound scaling multipliers (from paper Table 2):
  B0: depth=1.0, width=1.0, res=224
  B1: depth=1.1, width=1.0, res=240
  B2: depth=1.2, width=1.1, res=260
  B3: depth=1.4, width=1.2, res=300
  B4: depth=1.8, width=1.4, res=380
  B5: depth=2.2, width=1.6, res=456
  B6: depth=2.6, width=1.8, res=528
  B7: depth=3.1, width=2.0, res=600

Registered names
----------------
  efficientnet_b0  — ~5.3M params
  efficientnet_b1  — ~7.8M params
  efficientnet_b2  — ~9.1M params
  efficientnet_b3  — ~12M params
  efficientnet_b4  — ~19M params
  efficientnet_b5  — ~30M params
  efficientnet_b6  — ~43M params
  efficientnet_b7  — ~66M params

Note: This is the ORIGINAL EfficientNet (B-series).  EfficientNetV2 is a
separate architecture in efficientnet.py (uses FusedMBConv blocks).

Citation
--------
  Tan, M. & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for
  Convolutional Neural Networks. ICML. arXiv:1905.11946.
"""

import hale_core.nn.vision.cnn.efficientnet_b._helpers  # noqa: F401 - module-level registrations
import hale_core.nn.vision.cnn.efficientnet_b._variants  # noqa: F401 - module-level registrations
from hale_core.nn.vision.cnn.efficientnet_b.efficient_net_b import EfficientNetB
from hale_core.nn.vision.cnn.efficientnet_b.efficient_net_b0 import EfficientNetB0
from hale_core.nn.vision.cnn.efficientnet_b.efficient_net_b1 import EfficientNetB1
from hale_core.nn.vision.cnn.efficientnet_b.efficient_net_b2 import EfficientNetB2
from hale_core.nn.vision.cnn.efficientnet_b.efficient_net_b3 import EfficientNetB3
from hale_core.nn.vision.cnn.efficientnet_b.efficient_net_b4 import EfficientNetB4
from hale_core.nn.vision.cnn.efficientnet_b.efficient_net_b5 import EfficientNetB5
from hale_core.nn.vision.cnn.efficientnet_b.efficient_net_b6 import EfficientNetB6
from hale_core.nn.vision.cnn.efficientnet_b.efficient_net_b7 import EfficientNetB7
from hale_core.nn.vision.cnn.efficientnet_b.mb_conv_b import MBConvB
from hale_core.nn.vision.cnn.efficientnet_b.squeeze_excite_b import SqueezeExciteB
from hale_core.nn.vision.cnn.efficientnet_b.stochastic_depth import StochasticDepth

__all__ = [
    "EfficientNetB",
    "EfficientNetB0",
    "EfficientNetB1",
    "EfficientNetB2",
    "EfficientNetB3",
    "EfficientNetB4",
    "EfficientNetB5",
    "EfficientNetB6",
    "EfficientNetB7",
    "MBConvB",
    "SqueezeExciteB",
    "StochasticDepth",
]
