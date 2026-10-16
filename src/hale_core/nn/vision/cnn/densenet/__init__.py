"""
DenseNet Vision Backbones
=========================

Implements the DenseNet family (Huang et al. CVPR 2017, "Densely Connected
Convolutional Networks").

The key innovation: each layer is connected to ALL subsequent layers in its
dense block.  Instead of a single skip connection (ResNet), features are
concatenated — every layer has direct access to all prior feature maps,
maximising feature reuse and enabling very narrow layers (growth rate k=32)
to be competitive with much wider non-dense networks.

Architecture
------------
  Initial conv: Conv(3→2k, 7×7, s2, pad3) + BN + ReLU + MaxPool(3,s2)

  DenseBlock: each of L layers produces k new channels via:
    BN → ReLU → Conv1×1(in→4k) → BN → ReLU → Conv3×3(4k→k)
    Input to layer ℓ = concat of all prior layer outputs + initial input

  Transition: compresses and downsamples between blocks:
    BN → ReLU → Conv1×1(C→C//2) → AvgPool(2,s2)

  Four DenseBlocks separated by three Transitions, then:
    BN → ReLU → AdaptiveAvgPool → Linear → embed_dim

Feature map extraction: we return one feature map per DenseBlock output
(before the Transition's AvgPool), giving 4 spatial scales.

Registered names
----------------
  densenet121 — layers=(6,12,24,16), embed_dim=1024
  densenet169 — layers=(6,12,32,32), embed_dim=1664
  densenet201 — layers=(6,12,48,32), embed_dim=1920
  densenet264 — layers=(6,12,64,48), embed_dim=2688

Citation
--------
  Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017).
  Densely Connected Convolutional Networks. CVPR 2017. arXiv:1608.06993.
"""

import hale_core.nn.vision.cnn.densenet._helpers  # noqa: F401 - module-level registrations
from hale_core.nn.vision.cnn.densenet.dense_block import DenseBlock
from hale_core.nn.vision.cnn.densenet.dense_net import DenseNet
from hale_core.nn.vision.cnn.densenet.dense_net121 import DenseNet121
from hale_core.nn.vision.cnn.densenet.dense_net169 import DenseNet169
from hale_core.nn.vision.cnn.densenet.dense_net201 import DenseNet201
from hale_core.nn.vision.cnn.densenet.dense_net264 import DenseNet264

__all__ = ["DenseBlock", "DenseNet", "DenseNet121", "DenseNet169", "DenseNet201", "DenseNet264"]
