"""
ResNeSt Vision Backbones
=========================

Implements the ResNeSt family (Zhang et al. 2020, "ResNeSt: Split-Attention
Networks"). arXiv:2004.08955.

Architecture
------------
ResNeSt extends ResNet by replacing the 3×3 bottleneck convolution with a
**Split-Attention** convolution, which combines grouped convolutions with
attention across splits.

Key idea: split feature maps into `radix` branches, apply convolution to each,
then use a soft-max gating mechanism to recombine them:

  Input [B, C_in, H, W]
    ↓ Conv(C_in → out*radix, groups=groups)   # create radix feature splits
    ↓ reshape to [B, radix, C_out//groups, H, W] per group
    ↓ element-wise sum across radix → [B, C_out, H, W]
    ↓ GlobalAvgPool → [B, C_out, 1, 1]
    ↓ BN → Conv(C_out → C_out//4) → ReLU → Conv(C_out//4 → C_out*radix)
    ↓ reshape → softmax over radix axis → attention weights [B, radix, C_out//groups, 1, 1]
    ↓ weighted sum of splits → [B, C_out, H, W]

This is "attention across feature splits" — different from SE (channel attention
over pooled features) or CBAM (spatial + channel).  Split attention generalises
SKNet (Selective Kernel) to multiple groups.

ResNeSt Bottleneck: SplitAttentionConv2d replaces the 3×3 conv.
Stage structure (same as ResNet):
  Stem + 4 stages with depths depending on model size.

Feature map outputs:
  feature_maps[0] → [B, 256,  H/4,  W/4]
  feature_maps[1] → [B, 512,  H/8,  W/8]
  feature_maps[2] → [B, 1024, H/16, W/16]
  feature_maps[3] → [B, 2048, H/32, W/32]

Registered names
----------------
  resnest50    — depths=[3, 4, 6, 3],  ~27M params
  resnest101   — depths=[3, 4, 23, 3], ~48M params
  resnest200   — depths=[3, 24, 36, 3],~70M params
  resnest269   — depths=[3, 30, 48, 8],~110M params

Citation
--------
  Zhang, H. et al. (2020). ResNeSt: Split-Attention Networks. arXiv:2004.08955.
"""

from hale_core.nn.vision.cnn.resnest.res_ne_st import ResNeSt
from hale_core.nn.vision.cnn.resnest.res_ne_st50 import ResNeSt50
from hale_core.nn.vision.cnn.resnest.res_ne_st101 import ResNeSt101
from hale_core.nn.vision.cnn.resnest.res_ne_st200 import ResNeSt200
from hale_core.nn.vision.cnn.resnest.res_ne_st269 import ResNeSt269
from hale_core.nn.vision.cnn.resnest.res_ne_st_bottleneck import ResNeStBottleneck
from hale_core.nn.vision.cnn.resnest.split_attention_conv2d import SplitAttentionConv2d

__all__ = [
    "ResNeSt",
    "ResNeSt101",
    "ResNeSt200",
    "ResNeSt269",
    "ResNeSt50",
    "ResNeStBottleneck",
    "SplitAttentionConv2d",
]
