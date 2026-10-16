"""
ResNet + Deformable Convolution v2 (DCNv2) Backbones
=====================================================

Extends ResNet-50/101 by replacing the 3×3 convolutions in late stages
(stages 2 and 3) with Deformable Convolution v2 (Zhu et al. 2019).

Why deformable convolutions?
-----------------------------
Standard convolutions sample a fixed rectangular grid of input positions,
which is inefficient for objects at non-canonical scales, poses, or
orientations.  Deformable convolutions learn *per-position offsets* that
shift each sampling point, allowing the receptive field to adapt to the
geometry of the image content.

DCNv2 vs DCNv1
--------------
DCNv1 (Dai et al. 2017): learns 2D offsets per sampling point (Δx, Δy for
  each of the k² kernel positions).
DCNv2 (Zhu et al. 2019): additionally learns a *modulation scalar* per
  sampling point (a sigmoid gating weight in [0, 1]) that controls how much
  each sampled position contributes.  This gives the network more flexibility
  to suppress irrelevant context.

Implementation
--------------
This module implements DCNv2 from scratch using torch grid_sample, avoiding
the need for CUDA-compiled extensions like mmcv/torchvision deform_conv.

For each spatial location (i, j) and kernel position (p, q):
  1. An offset_conv (small 3×3 conv → 2 × k² channels) produces Δx, Δy.
  2. A mask_conv   (small 3×3 conv → k² channels)       produces weights ∈ (0,1).
  3. The sampled features are: sum over (p,q) of mask[p,q] * sample(x, i+p+Δx, j+q+Δy)
  4. Sampling uses bilinear interpolation via F.grid_sample.

The offset_conv is zero-initialized so the network starts as a regular conv
and learns deformability over time.

Stages using DCN
----------------
  stage2 (stride-16): 3×3 convolutions replaced with DeformConv2d
  stage3 (stride-32): 3×3 convolutions replaced with DeformConv2d

Stage 0 (stride-4) and stage 1 (stride-8) retain standard convolutions
because:
  - Low-level texture features are well-captured by regular grids.
  - DCN introduces more parameters and compute; keeping it to later stages
    is a good FLOPs/accuracy tradeoff.

Registered names
----------------
  resnet50_dcn  — ResNet-50 with DCNv2 in stages 2 & 3
  resnet101_dcn — ResNet-101 with DCNv2 in stages 2 & 3

Citation
--------
  Dai et al. (2017). Deformable Convolutional Networks. ICCV.
  Zhu et al. (2019). Deformable ConvNets v2: More Deformable, Better Results. CVPR.
"""

from hale_core.nn.vision.cnn.resnet_dcn.dcn_bottleneck import DCNBottleneck
from hale_core.nn.vision.cnn.resnet_dcn.deform_conv2d import DeformConv2d
from hale_core.nn.vision.cnn.resnet_dcn.res_net50_dcn import ResNet50DCN
from hale_core.nn.vision.cnn.resnet_dcn.res_net101_dcn import ResNet101DCN
from hale_core.nn.vision.cnn.resnet_dcn.res_net_dcn import ResNetDCN

__all__ = ["DCNBottleneck", "DeformConv2d", "ResNet101DCN", "ResNet50DCN", "ResNetDCN"]
