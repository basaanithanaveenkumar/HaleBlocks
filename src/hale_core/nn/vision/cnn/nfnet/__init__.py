"""
NFNet Vision Backbones
======================

Implements the Normalizer-Free Network family (Brock et al. 2021,
"High-Performance Large-Scale Image Recognition Without Normalization",
https://arxiv.org/abs/2102.06171).

Architecture
------------
NFNet achieves state-of-the-art accuracy without BatchNorm via three innovations:

  1. Scaled Weight Standardisation (SWS): normalises convolution weights at
     each forward pass so activation variance stays bounded, replacing BN.

  2. Variance Preservation in Residual Branches: the standard residual formula
     ``y = x + F(x)`` grows activation variance monotonically.  NFNet corrects
     this with ``y = x + alpha * F(x / beta)`` where alpha ≈ 0.2 (learned) and
     beta = sqrt(accumulated variance), keeping Var(y) ≈ Var(x).

  3. Adaptive Gradient Clipping (AGC): clips gradients by the ratio of the
     parameter norm to the gradient norm, ``g ← g * min(1, λ ||w|| / ||g||)``.
     This is a training-time utility; see ``agc_clip()`` below.

Block structure
---------------
  NFBlock (SE variant, group conv):
    WSConv 1×1 (pre-activation, /beta)
    → GELU
    → WSConv 3×3 groups=group_size (stride if downsample)
    → GELU
    → WSConv 1×1
    → SE (squeeze-excite, squeeze_ratio=0.5)
    → * alpha (learnable)
    → + residual (projected shortcut if needed)

Stage configs (widths, depths, group_size=128):
  F0: depths=[1,2,6,3],   widths=[256,512,1536,1536]
  F1: depths=[2,4,12,6],  widths=[256,512,1536,1536]
  F2: depths=[3,6,18,9],  widths=[256,512,1536,1536]
  F3: depths=[4,8,24,12], widths=[256,512,1536,1536]
  F4: depths=[5,10,30,15],widths=[256,512,1536,1536]
  F5: depths=[6,12,36,18],widths=[256,512,1536,1536]

Stem: 4 WSConv2d layers totalling 4× spatial reduction.

Registered names
----------------
  nfnet_f0 — depths=[1,2,6,3]
  nfnet_f1 — depths=[2,4,12,6]
  nfnet_f2 — depths=[3,6,18,9]
  nfnet_f3 — depths=[4,8,24,12]
  nfnet_f4 — depths=[5,10,30,15]
  nfnet_f5 — depths=[6,12,36,18]

Citation
--------
  Brock, A., De, S., Smith, S. L., & Simonyan, K. (2021).
  High-Performance Large-Scale Image Recognition Without Normalization. ICML.
  arXiv:2102.06171
"""

from hale_core.nn.vision.cnn.nfnet._helpers import agc_clip
from hale_core.nn.vision.cnn.nfnet.nf_block import NFBlock
from hale_core.nn.vision.cnn.nfnet.nf_net import NFNet
from hale_core.nn.vision.cnn.nfnet.nf_net_f0 import NFNetF0
from hale_core.nn.vision.cnn.nfnet.nf_net_f1 import NFNetF1
from hale_core.nn.vision.cnn.nfnet.nf_net_f2 import NFNetF2
from hale_core.nn.vision.cnn.nfnet.nf_net_f3 import NFNetF3
from hale_core.nn.vision.cnn.nfnet.nf_net_f4 import NFNetF4
from hale_core.nn.vision.cnn.nfnet.nf_net_f5 import NFNetF5
from hale_core.nn.vision.cnn.nfnet.squeeze_excite import SqueezeExcite
from hale_core.nn.vision.cnn.nfnet.stochastic_depth import StochasticDepth
from hale_core.nn.vision.cnn.nfnet.ws_conv2d import WSConv2d

__all__ = [
    "NFBlock",
    "NFNet",
    "NFNetF0",
    "NFNetF1",
    "NFNetF2",
    "NFNetF3",
    "NFNetF4",
    "NFNetF5",
    "SqueezeExcite",
    "StochasticDepth",
    "WSConv2d",
    "agc_clip",
]
