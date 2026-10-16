"""
SENet / SE-ResNet Vision Backbones
=====================================

Implements SE-ResNet variants (Hu et al. 2018, "Squeeze-and-Excitation Networks",
CVPR 2018, ILSVRC 2017 classification winner).

Reference: https://arxiv.org/abs/1709.01507

Architecture
------------
Squeeze-and-Excitation (SE) adds a channel attention mechanism to residual blocks.
The SE block recalibrates channel-wise feature responses by:

  1. Squeeze:   GlobalAvgPool [B, C, H, W] → [B, C, 1, 1]
                Collapses spatial information into a channel descriptor.
  2. Excitation: Two FC layers with a bottleneck:
                 FC(C→C//r) → ReLU → FC(C//r→C) → Sigmoid → [B, C, 1, 1]
                 Learns non-linear channel interdependencies.
  3. Scale:     x × scale — multiplies the input feature map channel-wise.

The SE block is inserted after the last conv in a Bottleneck block (before the
residual addition), giving the network "self-recalibration" ability at each stage.

Parameter overhead: 2C²/r per block.  For C=256, r=16: ~8192 extra params per block.

Stage structure (same as ResNet-50/101/152):
  Stem    : 7×7 Conv + BN + ReLU + MaxPool (s2+s2) → 1/4 resolution
  Stage 0 : SEBottleneck × [3, 3, 3], 64 planes,  stride=1  → [B, 256,  H/4, W/4]
  Stage 1 : SEBottleneck × [4, 4, 4], 128 planes, stride=2  → [B, 512,  H/8, W/8]
  Stage 2 : SEBottleneck × [6,23,36], 256 planes, stride=2  → [B, 1024, H/16,W/16]
  Stage 3 : SEBottleneck × [3, 3, 3], 512 planes, stride=2  → [B, 2048, H/32,W/32]

SE-ResNeXt variant: replaces the grouped 3×3 conv in the bottleneck
  (cardinality C=32, base_width=4 for seresnext50_32x4d).

Registered names
----------------
  se_resnet50       — SEBottleneck × [3, 4, 6, 3],  embed_dim=2048
  se_resnet101      — SEBottleneck × [3, 4, 23, 3], embed_dim=2048
  se_resnet152      — SEBottleneck × [3, 8, 36, 3], embed_dim=2048
  seresnext50_32x4d — SE-ResNeXt (cardinality=32, base_width=4), embed_dim=2048

Citation
--------
  Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks. CVPR.
  arXiv:1709.01507.
"""

from hale_core.nn.vision.cnn.senet.se_block import SEBlock
from hale_core.nn.vision.cnn.senet.se_bottleneck import SEBottleneck
from hale_core.nn.vision.cnn.senet.se_res_ne_xt50_32x4d import SEResNeXt50_32x4d
from hale_core.nn.vision.cnn.senet.se_res_net import SEResNet
from hale_core.nn.vision.cnn.senet.se_res_net50 import SEResNet50
from hale_core.nn.vision.cnn.senet.se_res_net101 import SEResNet101
from hale_core.nn.vision.cnn.senet.se_res_net152 import SEResNet152

__all__ = [
    "SEBlock",
    "SEBottleneck",
    "SEResNeXt50_32x4d",
    "SEResNet",
    "SEResNet101",
    "SEResNet152",
    "SEResNet50",
]
