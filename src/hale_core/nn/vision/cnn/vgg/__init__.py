"""
VGGNet Vision Backbones
=======================

Implements the VGG family (Simonyan & Zisserman, ICLR 2015).
VGG demonstrated that depth with small 3×3 convolutions outperforms shallower
networks with larger kernels — a key insight that shaped CNN design for years.

Architecture
------------
All VGG variants share the same pattern: sequential conv-pool stages followed
by a 3-layer FC classifier. The configs differ only in how many conv layers
appear per stage:

  Stage 0: 1–2 × Conv(64) + MaxPool   → H/2
  Stage 1: 1–2 × Conv(128) + MaxPool  → H/4
  Stage 2: 2–4 × Conv(256) + MaxPool  → H/8
  Stage 3: 2–4 × Conv(512) + MaxPool  → H/16
  Stage 4: 2–4 × Conv(512) + MaxPool  → H/32
  Classifier: AdaptiveAvgPool(7,7) → FC(4096) → FC(4096)

_bn variants add BatchNorm2d after every Conv, which wasn't in the original
paper but dramatically improves training stability and final accuracy.

Weight initialisation:
  Convolutions: Kaiming normal.
  Linear layers: truncated normal (std=0.01).

Registered names
----------------
  vgg11, vgg13, vgg16, vgg19           — without BatchNorm
  vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn — with BatchNorm

Citation
--------
  Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for
  Large-Scale Image Recognition. ICLR 2015. arXiv:1409.1556.
"""

import hale_core.nn.vision.cnn.vgg._helpers  # noqa: F401 - module-level registrations
from hale_core.nn.vision.cnn.vgg.vgg import VGG
from hale_core.nn.vision.cnn.vgg.vgg11 import VGG11
from hale_core.nn.vision.cnn.vgg.vgg11_bn import VGG11BN
from hale_core.nn.vision.cnn.vgg.vgg13 import VGG13
from hale_core.nn.vision.cnn.vgg.vgg13_bn import VGG13BN
from hale_core.nn.vision.cnn.vgg.vgg16 import VGG16
from hale_core.nn.vision.cnn.vgg.vgg16_bn import VGG16BN
from hale_core.nn.vision.cnn.vgg.vgg19 import VGG19
from hale_core.nn.vision.cnn.vgg.vgg19_bn import VGG19BN

__all__ = ["VGG", "VGG11", "VGG11BN", "VGG13", "VGG13BN", "VGG16", "VGG16BN", "VGG19", "VGG19BN"]
