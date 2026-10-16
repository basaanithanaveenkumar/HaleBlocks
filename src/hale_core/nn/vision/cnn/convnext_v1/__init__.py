"""
ConvNeXt V1 Vision Backbones
=============================

Implements the ConvNeXt V1 family (Liu et al. 2022, "A ConvNet for the 2020s",
CVPR 2022, https://arxiv.org/abs/2201.03545).

Design philosophy
-----------------
ConvNeXt V1 is a pure CNN redesigned to systematically match Swin Transformer.
Starting from ResNet-50, the authors apply a sequence of design improvements:

  1. Patchify stem: 4×4 stride-4 conv (no 7×7 + maxpool).
  2. Stage compute ratio: [3,3,9,3] matching Swin's stage distribution.
  3. Inverted bottleneck: expand channels 4× inside each block (like MobileNet).
  4. Depthwise 7×7 conv: matches Swin's 7×7 local window receptive field.
  5. Fewer activations: only one GELU (after the expansion linear).
  6. Fewer normalisations: only one LayerNorm per block (before the linear layers).
  7. LayerNorm instead of BatchNorm: matches transformer-style normalisation.
  8. Separate downsampling: LayerNorm + 2×2 stride-2 conv between stages.
  9. Layer Scale: learnable per-channel scalar γ (init 1e-6) stabilises training
     of deep networks; enabled for large/xlarge.

Block structure (ConvNeXtV1Block)
----------------------------------
  input [B, C, H, W]  (channels-first)
    → DWConv(C, 7×7, groups=C)
    → permute to [B, H, W, C]
    → LayerNorm(C, ε=1e-6)
    → Linear(C → 4C)
    → GELU
    → Linear(4C → C)
    → * layer_scale_gamma  (if layer_scale_init_value > 0)
    → permute back to [B, C, H, W]
    → stochastic depth + residual

Difference from V2
------------------
  V1: no GRN (Global Response Normalization).
  V2: adds GRN after the wide (4C) GELU layer.
  V1: optional Layer Scale (per-channel scalar).
  V2: replaces Layer Scale with GRN.

Registered names
----------------
  convnext_v1_tiny   — depths=[3,3,9,3],   dim=96
  convnext_v1_small  — depths=[3,3,27,3],  dim=96
  convnext_v1_base   — depths=[3,3,27,3],  dim=128
  convnext_v1_large  — depths=[3,3,27,3],  dim=192  (layer_scale enabled)
  convnext_v1_xlarge — depths=[3,3,27,3],  dim=256  (layer_scale enabled)

Citations
---------
  Liu, Z., Mao, H., Wu, C.-Y., Feichtenhofer, C., Darrell, T., & Xie, S. (2022).
  A ConvNet for the 2020s. CVPR.  arXiv:2201.03545

  Woo, S. et al. (2023). ConvNeXt V2: Co-designing and Scaling ConvNets with
  Masked Autoencoders. CVPR.  arXiv:2301.00808
"""

from hale_core.nn.vision.cnn.convnext_v1.conv_ne_xt_v1 import ConvNeXtV1
from hale_core.nn.vision.cnn.convnext_v1.conv_ne_xt_v1_base import ConvNeXtV1Base
from hale_core.nn.vision.cnn.convnext_v1.conv_ne_xt_v1_block import ConvNeXtV1Block
from hale_core.nn.vision.cnn.convnext_v1.conv_ne_xt_v1_large import ConvNeXtV1Large
from hale_core.nn.vision.cnn.convnext_v1.conv_ne_xt_v1_small import ConvNeXtV1Small
from hale_core.nn.vision.cnn.convnext_v1.conv_ne_xt_v1_tiny import ConvNeXtV1Tiny
from hale_core.nn.vision.cnn.convnext_v1.conv_ne_xt_v1_x_large import ConvNeXtV1XLarge
from hale_core.nn.vision.cnn.convnext_v1.layer_norm2d import LayerNorm2d
from hale_core.nn.vision.cnn.convnext_v1.stochastic_depth import StochasticDepth

__all__ = [
    "ConvNeXtV1",
    "ConvNeXtV1Base",
    "ConvNeXtV1Block",
    "ConvNeXtV1Large",
    "ConvNeXtV1Small",
    "ConvNeXtV1Tiny",
    "ConvNeXtV1XLarge",
    "LayerNorm2d",
    "StochasticDepth",
]
