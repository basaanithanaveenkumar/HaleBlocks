"""
ConvNeXt V2 Vision Backbones
=============================

Implements the ConvNeXt V2 family (Woo et al. 2023, "ConvNeXt V2: Co-designing and
Scaling ConvNets with Masked Autoencoders").

Architecture evolution
----------------------
ConvNeXt V1 (Liu et al. 2022, "A ConvNet for the 2020s"):
  - Modernised ConvNet: depthwise 7×7, inverted bottleneck, GeLU, LayerNorm.
  - Matched ViT in accuracy and design style while remaining a pure CNN.

ConvNeXt V2 adds:
  - FCMAE (Fully Convolutional Masked Autoencoder) pre-training paradigm.
  - Global Response Normalization (GRN): a channel feature competition mechanism
    analogous to SE (Squeeze-Excite) but operating globally on each channel.

ConvNeXtV2Block
---------------
  Input [B, C, H, W]
  1. Depthwise 7×7 conv (groups=C) — captures long-range spatial structure.
     Channels-first → channels-last transpose for LayerNorm efficiency.
  2. LayerNorm (channels-last, ε=1e-6).
  3. Pointwise Linear expand: C → 4C.
  4. GeLU activation.
  5. Global Response Normalization (GRN):
       x_norm = ‖x‖_2 across spatial dims → [B, 1, 1, C]
       x = x * (x_norm / mean(x_norm)) + bias   (gated by spatial norms)
     This penalises redundant channels and encourages diverse feature activations.
  6. Pointwise Linear contract: 4C → C.
  Residual add.

Stage downsampling
------------------
  Stem: 4×4 conv stride 4 → LayerNorm  (initial patch-like tokenization)
  Downsample between stages: LayerNorm + 2×2 conv stride 2.

Registered names and configs
-----------------------------
  convnext_tiny   — depths=[3,3,9,3],    dims=[96,192,384,768]
  convnext_small  — depths=[3,3,27,3],   dims=[96,192,384,768]
  convnext_base   — depths=[3,3,27,3],   dims=[128,256,512,1024]
  convnext_large  — depths=[3,3,27,3],   dims=[192,384,768,1536]
  convnext_xlarge — depths=[3,3,27,3],   dims=[256,512,1024,2048]
  convnext_huge   — depths=[3,3,27,3],   dims=[352,704,1408,2816]

Citations
---------
  Liu et al. (2022). A ConvNet for the 2020s. CVPR.
  Woo et al. (2023). ConvNeXt V2: Co-designing and Scaling ConvNets with MAE. CVPR.
"""

from hale_core.nn.vision.cnn.convnext.conv_ne_xt_base import ConvNeXtBase
from hale_core.nn.vision.cnn.convnext.conv_ne_xt_huge import ConvNeXtHuge
from hale_core.nn.vision.cnn.convnext.conv_ne_xt_large import ConvNeXtLarge
from hale_core.nn.vision.cnn.convnext.conv_ne_xt_small import ConvNeXtSmall
from hale_core.nn.vision.cnn.convnext.conv_ne_xt_tiny import ConvNeXtTiny
from hale_core.nn.vision.cnn.convnext.conv_ne_xt_v2 import ConvNeXtV2
from hale_core.nn.vision.cnn.convnext.conv_ne_xt_v2_block import ConvNeXtV2Block
from hale_core.nn.vision.cnn.convnext.conv_ne_xt_x_large import ConvNeXtXLarge
from hale_core.nn.vision.cnn.convnext.grn import GRN
from hale_core.nn.vision.cnn.convnext.layer_norm2d import LayerNorm2d
from hale_core.nn.vision.cnn.convnext.stochastic_depth import StochasticDepth

__all__ = [
    "ConvNeXtBase",
    "ConvNeXtHuge",
    "ConvNeXtLarge",
    "ConvNeXtSmall",
    "ConvNeXtTiny",
    "ConvNeXtV2",
    "ConvNeXtV2Block",
    "ConvNeXtXLarge",
    "GRN",
    "LayerNorm2d",
    "StochasticDepth",
]
