"""
CoAtNet — Marrying Convolution and Attention (Dai et al., 2021)
===============================================================

Reference: "CoAtNet: Marrying Convolution and Attention for All Data Sizes"
           https://arxiv.org/abs/2106.04803

Architecture overview
---------------------
CoAtNet is a hybrid architecture that uses convolution in early stages
(where local patterns matter) and self-attention in later stages
(where global context matters). This staged hybrid outperforms pure
ViT on small/medium data while matching it on large data.

Stage layout (5 stages total):
  S0 (Conv Stem): 2× Conv2d stride-2, pure convolution, H→H/4
  S1 (MBConv):    Mobile Inverted Bottleneck blocks, H/4→H/8
  S2 (MBConv):    Wider/deeper MBConv, H/8→H/16
  S3 (RelAttn):   Transformer blocks with relative position bias, H/16→H/32
  S4 (RelAttn):   Deeper transformer, H/32 (final)

Feature maps for FPN come from S1..S4 (4 multi-scale outputs).

Key design decisions
--------------------
**MBConv block** (Mobile Inverted Bottleneck):
  - Pre-BatchNorm: BN is applied BEFORE the block (normalise input),
    unlike standard post-norm in the original MobileNetV2.
  - Depthwise separable: expand 1×1 → depthwise 3×3 → SE → project 1×1
  - Squeeze-Excitation (SE): channel attention using global avg pool + 2-FC.
  - Residual skip only when input/output dims match.

**Relative Multi-Head Attention block**:
  - Pre-LayerNorm for attention AND FFN sublayers.
  - Relative position bias: a table of shape [2*Hmax-1, 2*Wmax-1] is learned
    for each head; biases are added to attention logits based on (row_diff, col_diff).
  - Bias indices are clipped so the table covers all positions up to max_img_size.
  - Output: standard Q·K^T/sqrt(d) + rel_bias, then V aggregation.

**Downsampling** between stages:
  - MaxPool2d(2×2) for spatial halving.
  - 1×1 Conv to change channel dimension.

BackboneOutput
--------------
  pooled_output : [B, dims[-1]]                  — global average pool of S4
  feature_maps  : [S1, S2, S3, S4]               — 4 multi-scale spatial maps
  patch_tokens  : None                           — not applicable
"""

import hale_core.nn.vision.hybrid.coatnet._helpers  # noqa: F401 - module-level registrations
from hale_core.nn.vision.hybrid.coatnet.co_at_net0 import CoAtNet0
from hale_core.nn.vision.hybrid.coatnet.co_at_net1 import CoAtNet1
from hale_core.nn.vision.hybrid.coatnet.co_at_net2 import CoAtNet2
from hale_core.nn.vision.hybrid.coatnet.co_at_net3 import CoAtNet3
from hale_core.nn.vision.hybrid.coatnet.co_at_net7 import CoAtNet7
from hale_core.nn.vision.hybrid.coatnet.co_at_net_attn_stage import CoAtNetAttnStage
from hale_core.nn.vision.hybrid.coatnet.co_at_net_backbone import CoAtNetBackbone
from hale_core.nn.vision.hybrid.coatnet.co_at_net_conv_stage import CoAtNetConvStage
from hale_core.nn.vision.hybrid.coatnet.drop_path import DropPath
from hale_core.nn.vision.hybrid.coatnet.mb_conv_block import MBConvBlock
from hale_core.nn.vision.hybrid.coatnet.relative_attention_block import RelativeAttentionBlock
from hale_core.nn.vision.hybrid.coatnet.relative_position_bias import RelativePositionBias
from hale_core.nn.vision.hybrid.coatnet.squeeze_excitation import SqueezeExcitation

__all__ = [
    "CoAtNet0",
    "CoAtNet1",
    "CoAtNet2",
    "CoAtNet3",
    "CoAtNet7",
    "CoAtNetAttnStage",
    "CoAtNetBackbone",
    "CoAtNetConvStage",
    "DropPath",
    "MBConvBlock",
    "RelativeAttentionBlock",
    "RelativePositionBias",
    "SqueezeExcitation",
]
