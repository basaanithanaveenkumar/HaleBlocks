"""
EfficientNetV2 Vision Backbones (Tan & Le 2021)
=================================================

Reference: "EfficientNetV2: Smaller Models and Faster Training"
           https://arxiv.org/abs/2104.00298

Architecture Overview
---------------------
EfficientNetV2 builds on EfficientNet by replacing the initial MBConv blocks
with Fused-MBConv blocks, which are faster on modern accelerators.

Two block types:
  MBConv (Mobile Inverted Bottleneck Conv):
    input → expand 1×1 (ratio r) → depthwise 3×3 or 5×5 → SE → project 1×1
    Used in later stages where depthwise is more efficient.

  FusedMBConv:
    input → fused_expand 3×3 (ratio r, no depthwise separation) → SE → project 1×1
    Used in early stages where tensor sizes are large (channel count small).
    Fusing expand+depthwise into a single 3×3 avoids the bandwidth bottleneck
    of large input feature maps.

Squeeze-Excitation (SE):
  Global avg pool → fc1 (C//se_ratio) → SiLU → fc2 (C) → Sigmoid → multiply input.
  Recalibrates channel-wise feature responses adaptively.

Stochastic Depth (DropPath):
  Randomly drops entire residual branches during training (survival probability
  linearly interpolated from 1.0 at layer 0 to (1 - drop_path_rate) at last layer).
  Acts as a regularizer and improves accuracy.

Stage structure (for S variant):
  Stage 0: FusedMBConv, r=1, k=3, stride=1, channels=24,  n=2
  Stage 1: FusedMBConv, r=4, k=3, stride=2, channels=48,  n=4
  Stage 2: FusedMBConv, r=4, k=3, stride=2, channels=64,  n=4
  Stage 3: MBConv,      r=4, k=3, stride=2, channels=128, n=6
  Stage 4: MBConv,      r=6, k=3, stride=1, channels=160, n=9
  Stage 5: MBConv,      r=6, k=3, stride=2, channels=256, n=15
  Head:    Conv1×1 → 1280-dim

Feature pyramid:
  feature_maps[0] = stride-4 output (after stem)
  feature_maps[1] = stride-8 output (after stage 1 down)
  feature_maps[2] = stride-16 output (after stage 2 down + stage 3)
  feature_maps[3] = stride-32 output (after stage 5)

Registered variants:
  efficientnet_v2_s   — small
  efficientnet_v2_m   — medium
  efficientnet_v2_l   — large
  efficientnet_v2_xl  — extra-large
"""

import hale_core.nn.vision.cnn.efficientnet._helpers  # noqa: F401 - module-level registrations
from hale_core.nn.vision.cnn.efficientnet.efficient_net_v2_backbone import EfficientNetV2Backbone
from hale_core.nn.vision.cnn.efficientnet.efficient_net_v2_l import EfficientNetV2L
from hale_core.nn.vision.cnn.efficientnet.efficient_net_v2_m import EfficientNetV2M
from hale_core.nn.vision.cnn.efficientnet.efficient_net_v2_s import EfficientNetV2S
from hale_core.nn.vision.cnn.efficientnet.efficient_net_v2_xl import EfficientNetV2XL
from hale_core.nn.vision.cnn.efficientnet.fused_mb_conv import FusedMBConv
from hale_core.nn.vision.cnn.efficientnet.mb_conv import MBConv
from hale_core.nn.vision.cnn.efficientnet.squeeze_excite import SqueezeExcite
from hale_core.nn.vision.cnn.efficientnet.stochastic_depth import StochasticDepth

__all__ = [
    "EfficientNetV2Backbone",
    "EfficientNetV2L",
    "EfficientNetV2M",
    "EfficientNetV2S",
    "EfficientNetV2XL",
    "FusedMBConv",
    "MBConv",
    "SqueezeExcite",
    "StochasticDepth",
]
