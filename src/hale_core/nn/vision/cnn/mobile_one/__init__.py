"""
MobileOne — Re-Parameterisable Efficient Backbone (Vasu et al. 2022, Apple Research)
======================================================================================

Reference: "MobileOne: An Improved One millisecond Mobile Backbone"
           https://arxiv.org/abs/2206.04040

Key Idea: Structural Re-Parameterisation
-----------------------------------------
At TRAINING time, each MobileOne block uses MULTIPLE parallel branches:
  - k branches of depthwise 3×3 conv (each with BN)
  - k branches of 1×1 pointwise conv (each with BN)
  - 1 identity branch (BN only, if in_channels == out_channels and stride == 1)

At INFERENCE time, all branches are FUSED into a SINGLE 3×3 conv + bias,
achieving the same output as all branches combined — but with zero overhead.

This gives the expressiveness of multi-branch training with the efficiency
of single-conv inference.

Reparameterisation formula:
  For each branch (weight W_i, BN: γ, β, μ, σ²), the equivalent conv is:
    W_eq = γ/σ · W_i
    b_eq = β - γ·μ/σ
  All branches are summed: W = Σ W_eq_i, b = Σ b_eq_i

  For the identity branch: W is a 1-in-1-out identity kernel (eye for depthwise,
  pointwise identity), then merged as above.

Why this works:
  BN(Conv(x)) = γ/σ · Conv(x) + (β - γμ/σ)
  Sum of linear ops (BN∘Conv) with same input = single linear op.
  So the entire k-branch structure can be collapsed to one conv.

Variants (s0..s4)
-----------------
  s0: width_multiplier=[0.75, 1.0, 1.0, 2.0], k=4  — smallest, fast on edge
  s1: width_multiplier=[1.5,  1.5, 2.0, 2.5], k=4  — balanced
  s2: width_multiplier=[1.5,  2.0, 2.5, 4.0], k=4  — higher accuracy
  s3: width_multiplier=[2.0,  2.5, 3.0, 4.0], k=4  — large
  s4: width_multiplier=[3.0,  3.5, 3.5, 4.0], k=4  — largest

After reparametrize(), model is in inference mode (single conv per block).
"""

import hale_core.nn.vision.cnn.mobile_one._helpers  # noqa: F401 - module-level registrations
from hale_core.nn.vision.cnn.mobile_one.mobile_one_backbone import MobileOneBackbone
from hale_core.nn.vision.cnn.mobile_one.mobile_one_block import MobileOneBlock
from hale_core.nn.vision.cnn.mobile_one.mobile_one_s0 import MobileOneS0
from hale_core.nn.vision.cnn.mobile_one.mobile_one_s1 import MobileOneS1
from hale_core.nn.vision.cnn.mobile_one.mobile_one_s2 import MobileOneS2
from hale_core.nn.vision.cnn.mobile_one.mobile_one_s3 import MobileOneS3
from hale_core.nn.vision.cnn.mobile_one.mobile_one_s4 import MobileOneS4

__all__ = [
    "MobileOneBackbone",
    "MobileOneBlock",
    "MobileOneS0",
    "MobileOneS1",
    "MobileOneS2",
    "MobileOneS3",
    "MobileOneS4",
]
