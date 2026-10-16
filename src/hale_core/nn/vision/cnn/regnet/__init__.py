"""
RegNet Vision Backbones
=======================

Implements the RegNet family (Radosavovic et al. 2020, "Designing Network Design
Spaces", CVPR 2020, https://arxiv.org/abs/2003.13678).

Key idea
--------
RegNet is not a single architecture but a principled *design space* for CNNs.
The authors analyse a large random population of AnyNet models and discover that
the best architectures share a simple structural regularity:

  - Widths grow approximately as a geometric sequence: w_j ≈ w_0 * α^j.
  - Depths, once quantised, produce a "RegNet" model (quantised design space).
  - A single group width g is shared across all stages.
  - Bottleneck ratio b=1 (no bottleneck; input and output widths match).

Two sub-families:
  RegNetX — no Squeeze-Excite (plain grouped bottleneck).
  RegNetY — with Squeeze-Excite (channel attention), typically ~0.5–1% better.

Block structure (ResBottleneckBlock)
-------------------------------------
  Conv 1×1 + BN + ReLU
  GroupConv 3×3 (groups = w / g) + BN + ReLU
  [SE if RegNetY]
  Conv 1×1 + BN
  + residual (projected shortcut if shapes differ)
  ReLU

Stem
----
  Conv(3→32, 3×3, stride=2) + BN + ReLU  → H/2 × W/2

Stages: stride-2 downsampling at the first block of each stage (from stage 1).

Registered names (GFlops × 100 suffix)
-----------------------------------------
  regnetx_002 … regnetx_320   (10 variants, no SE)
  regnety_002 … regnety_320   (10 variants, with SE)

Citation
--------
  Radosavovic, I., Kosaraju, R. P., Girshick, R., He, K., & Dollar, P. (2020).
  Designing Network Design Spaces. CVPR.
  arXiv:2003.13678
"""

import hale_core.nn.vision.cnn.regnet._helpers  # noqa: F401 - module-level registrations
import hale_core.nn.vision.cnn.regnet._variants  # noqa: F401 - module-level registrations
from hale_core.nn.vision.cnn.regnet.reg_net import RegNet
from hale_core.nn.vision.cnn.regnet.res_bottleneck_block import ResBottleneckBlock
from hale_core.nn.vision.cnn.regnet.squeeze_excite import SqueezeExcite

__all__ = ["RegNet", "ResBottleneckBlock", "SqueezeExcite"]
