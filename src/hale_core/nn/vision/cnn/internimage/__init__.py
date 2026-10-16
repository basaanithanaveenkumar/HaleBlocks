"""
InternImage — DCNv3-based Vision Backbone (Wang et al. 2022)
=============================================================

Reference: "InternImage: Exploring Large-Scale Vision Foundation Models
           with Deformable Convolutions"
           https://arxiv.org/abs/2211.05778

Key Innovation: DCNv3 (Deformable Convolution v3)
--------------------------------------------------
DCNv3 makes deformable convolution efficient and scalable by redesigning it
as a SEPARABLE operation, analogous to the ViT attention mechanism:

Standard DCNv2 (each of G groups has its own fully-connected projection):
  - Output = Σ_g Σ_k w_g,k · x(p + Δp_g,k) · m_g,k
  - Projection per group: O(C²) cost

DCNv3 (shared depthwise projection, separate per-group offsets/masks):
  - Depthwise over all channels first (shared, O(C) cost)
  - Then per-group offset/mask prediction (O(C) total, not O(C²))
  - Output = Σ_g Σ_k (W_proj · x(p + Δp_g,k)) · m_g,k
  - This makes DCNv3 as scalable as ViT attention

Deformable Convolution Sampling:
  Instead of a fixed k×k grid, DCNv3 learns offsets Δp_g,k for each
  of K sampling points in each of G groups. The network learns WHERE
  to look in the input feature map for each output location.
  - Input position p: any spatial location in the feature map
  - Learned offset Δp_g,k: 2D offset for sampling point k in group g
  - Sampling: x(p + Δp_g,k) via bilinear interpolation
  - Modulation mask m_g,k: scalar in [0,1] weighting each sample point

  This gives adaptive receptive fields: for cars, the network might learn
  to aggregate features from the wheels and roof; for text, each character
  might focus on adjacent strokes.

DCNv3 Layer Implementation (without the C extension):
  We implement DCNv3 using F.grid_sample (bilinear interpolation) on
  GPU-computed offset coordinates. This is slightly slower than the CUDA
  kernel but avoids the build dependency on DCNv3's custom ops.

Architecture
------------
  Stem: 4×4 stride-4 conv (patch-like), LayerNorm
  4 stages, each with:
    - N InternImage blocks (DCNv3 + FFN)
    - Downsampling layer between stages (2×2 stride-2 conv, LayerNorm)

  InternImageBlock:
    input → LayerNorm → DCNv3 → LayerNorm → FFN → residual add

  FFN: Linear → GELU → Linear (4× expand)

Registered variants:
  internimage_tiny:  C=64,  depths=[4,4,18,4],  groups=[4,8,16,32]
  internimage_small: C=80,  depths=[4,4,21,4],  groups=[5,10,20,40]
  internimage_base:  C=112, depths=[4,4,21,4],  groups=[7,14,28,56]
  internimage_large: C=160, depths=[5,5,22,5],  groups=[10,20,40,80]
  internimage_huge:  C=320, depths=[6,6,32,6],  groups=[20,40,80,160]
"""

import hale_core.nn.vision.cnn.internimage._helpers  # noqa: F401 - module-level registrations
from hale_core.nn.vision.cnn.internimage.dc_nv3 import DCNv3
from hale_core.nn.vision.cnn.internimage.intern_image_backbone import InternImageBackbone
from hale_core.nn.vision.cnn.internimage.intern_image_base import InternImageBase
from hale_core.nn.vision.cnn.internimage.intern_image_block import InternImageBlock
from hale_core.nn.vision.cnn.internimage.intern_image_downsample import InternImageDownsample
from hale_core.nn.vision.cnn.internimage.intern_image_ffn import InternImageFFN
from hale_core.nn.vision.cnn.internimage.intern_image_huge import InternImageHuge
from hale_core.nn.vision.cnn.internimage.intern_image_large import InternImageLarge
from hale_core.nn.vision.cnn.internimage.intern_image_small import InternImageSmall
from hale_core.nn.vision.cnn.internimage.intern_image_tiny import InternImageTiny

__all__ = [
    "DCNv3",
    "InternImageBackbone",
    "InternImageBase",
    "InternImageBlock",
    "InternImageDownsample",
    "InternImageFFN",
    "InternImageHuge",
    "InternImageLarge",
    "InternImageSmall",
    "InternImageTiny",
]
