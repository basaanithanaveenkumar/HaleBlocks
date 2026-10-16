"""
AlexNet Vision Backbone
=======================

Implements AlexNet (Krizhevsky, Sutskever & Hinton, NeurIPS 2012).
The first deep CNN to win ImageNet by a large margin, kickstarting the deep
learning revolution in computer vision.

Architecture
------------
Five convolutional stages followed by three fully-connected layers:

  Stage 0: Conv(3→64, 11×11, s4) + ReLU + MaxPool(3,s2)    → H/8
  Stage 1: Conv(64→192, 5×5, pad2) + ReLU + MaxPool(3,s2)  → H/16
  Stage 2: Conv(192→384, 3×3, pad1) + ReLU                 → H/16
  Stage 3: Conv(384→256, 3×3, pad1) + ReLU                 → H/16
  Stage 4: Conv(256→256, 3×3, pad1) + ReLU + MaxPool(3,s2) → H/32
  Classifier: AdaptiveAvgPool → Flatten → FC(4096) → FC(4096)

The AdaptiveAvgPool normalises spatial size before the FC classifier,
making the backbone work on any input resolution (original assumed 227×227).

Weight initialisation:
  Convolutions: Kaiming normal.
  Linear layers: truncated normal (std=0.01) — original used normal(0, 0.01).

Registered names
----------------
  alexnet — embed_dim=4096

Citation
--------
  Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet Classification
  with Deep Convolutional Neural Networks. NeurIPS, 25, 1097–1105.
"""

from hale_core.nn.vision.cnn.alexnet.alex_net import AlexNet
from hale_core.nn.vision.cnn.alexnet.alex_net_backbone import AlexNetBackbone

__all__ = ["AlexNet", "AlexNetBackbone"]
