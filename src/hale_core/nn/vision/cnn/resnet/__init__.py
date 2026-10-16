"""
ResNet Vision Backbones
=======================

Implements the full ResNet family (He et al. 2016, "Deep Residual Learning for Image
Recognition", https://arxiv.org/abs/1512.03385).

Architecture
------------
ResNet is built from stacked residual blocks of two designs:

  BasicBlock (used in ResNet-18 and ResNet-34):
    3×3 conv → BN → ReLU → 3×3 conv → BN → (+residual) → ReLU
    Expansion factor = 1 (output channels == input channels).

  Bottleneck (used in ResNet-50/101/152/200):
    1×1 conv (reduce) → BN → ReLU
    3×3 conv          → BN → ReLU
    1×1 conv (expand) → BN → (+residual) → ReLU
    Expansion factor = 4 (output channels = planes × 4).

Stage structure:
  Stem    : 7×7 conv, stride 2 → BN → ReLU → MaxPool stride 2  → 1/4 resolution
  Stage 0 : 64  channels, stride 1                               → 1/4 resolution
  Stage 1 : 128 channels, stride 2                               → 1/8 resolution
  Stage 2 : 256 channels, stride 2                               → 1/16 resolution
  Stage 3 : 512 channels, stride 2                               → 1/32 resolution

Weight initialisation:
  Convolutions: Kaiming normal (He init) for correct variance under ReLU.
  BatchNorm:    weight=1, bias=0 (standard identity init).
  Zero-init of the last BN in each residual branch (Goyal et al. "Accurate, Large
  Minibatch SGD"), which improves convergence stability at large batch sizes.

Registered names
----------------
  resnet18   — BasicBlock × [2, 2, 2, 2],  embed_dim=512
  resnet34   — BasicBlock × [3, 4, 6, 3],  embed_dim=512
  resnet50   — Bottleneck × [3, 4, 6, 3],  embed_dim=2048
  resnet101  — Bottleneck × [3, 4, 23, 3], embed_dim=2048
  resnet152  — Bottleneck × [3, 8, 36, 3], embed_dim=2048
  resnet200  — Bottleneck × [3, 24, 36, 3],embed_dim=2048

Citation
--------
  He et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
  Goyal et al. (2017). Accurate, Large Minibatch SGD. arXiv:1706.02677.
"""

from hale_core.nn.vision.cnn.resnet.basic_block import BasicBlock
from hale_core.nn.vision.cnn.resnet.bottleneck import Bottleneck
from hale_core.nn.vision.cnn.resnet.res_net import ResNet
from hale_core.nn.vision.cnn.resnet.res_net18 import ResNet18
from hale_core.nn.vision.cnn.resnet.res_net34 import ResNet34
from hale_core.nn.vision.cnn.resnet.res_net50 import ResNet50
from hale_core.nn.vision.cnn.resnet.res_net101 import ResNet101
from hale_core.nn.vision.cnn.resnet.res_net152 import ResNet152
from hale_core.nn.vision.cnn.resnet.res_net200 import ResNet200

__all__ = [
    "BasicBlock",
    "Bottleneck",
    "ResNet",
    "ResNet101",
    "ResNet152",
    "ResNet18",
    "ResNet200",
    "ResNet34",
    "ResNet50",
]
