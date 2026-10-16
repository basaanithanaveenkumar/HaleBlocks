"""
Inception / GoogLeNet Vision Backbones
=======================================

Implements GoogLeNet / Inception-v1 (Szegedy et al. CVPR 2015) and a clean
modern Inception-v3 (Szegedy et al. CVPR 2016) without auxiliary classifiers.

The key innovation is the Inception module: instead of choosing one filter size,
apply multiple in parallel and concatenate the results.  This captures features
at multiple scales simultaneously within a single layer.

Architecture — GoogLeNet / Inception-v1
-----------------------------------------
  Stem:    Conv(7×7,s2)+BN+ReLU → MaxPool → LRN → Conv(1×1) → Conv(3×3)+BN → MaxPool
  Stage 1: Inception 3a + 3b → MaxPool(3,s2)   → 480 ch, H/16
  Stage 2: Inception 4a–4e   → MaxPool(3,s2)   → 832 ch, H/32
  Stage 3: Inception 5a + 5b → GlobalAvgPool   → [B, 1024]

Architecture — Inception-v3 (clean version, no auxiliary classifiers)
-----------------------------------------------------------------------
  Uses factorised convolutions throughout:
  Type-A: replaces 5×5 with two 3×3
  Type-B: factorises n×n into 1×n + n×1 (asymmetric)
  Type-C: splits 3×3 into 1×3 + 3×1 in parallel (for high-level stages)
  embed_dim=1280.

Registered names
----------------
  inception_v1, inception_v3

Citation
--------
  Szegedy, C., Liu, W., Jia, Y., et al. (2015). Going Deeper with Convolutions.
  CVPR 2015. arXiv:1409.4842.

  Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016).
  Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
  arXiv:1512.00567.
"""

import hale_core.nn.vision.cnn.inception._inception_type_a  # noqa: F401 - module-level registrations
import hale_core.nn.vision.cnn.inception._inception_type_b  # noqa: F401 - module-level registrations
import hale_core.nn.vision.cnn.inception._inception_type_c  # noqa: F401 - module-level registrations
import hale_core.nn.vision.cnn.inception._reduction_a  # noqa: F401 - module-level registrations
import hale_core.nn.vision.cnn.inception._reduction_b  # noqa: F401 - module-level registrations
from hale_core.nn.vision.cnn.inception.conv_bn_relu import ConvBnRelu
from hale_core.nn.vision.cnn.inception.goog_le_net import GoogLeNet
from hale_core.nn.vision.cnn.inception.inception_module_v1 import InceptionModuleV1
from hale_core.nn.vision.cnn.inception.inception_v1_backbone import InceptionV1Backbone
from hale_core.nn.vision.cnn.inception.inception_v3 import InceptionV3
from hale_core.nn.vision.cnn.inception.inception_v3_backbone import InceptionV3Backbone

__all__ = [
    "ConvBnRelu",
    "GoogLeNet",
    "InceptionModuleV1",
    "InceptionV1Backbone",
    "InceptionV3",
    "InceptionV3Backbone",
]
