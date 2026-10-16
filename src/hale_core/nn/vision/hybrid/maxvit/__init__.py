"""
MaxViT — Multi-Axis Vision Transformer (Tu et al., 2022)
=========================================================

Reference: "MaxViT: Multi-Axis Vision Transformer"
           https://arxiv.org/abs/2204.01697

Architecture overview
---------------------
MaxViT solves the O(N²) attention problem for high-resolution images by
interleaving two complementary attention patterns in every block:

1. **Local Block Attention** (window / "block" attention):
   Partitions the spatial map into non-overlapping windows of size P×P
   (e.g. 8×8). Self-attention runs inside each window independently.
   → Captures fine-grained local patterns.

2. **Dilated Grid Attention** (global / "grid" attention):
   Instead of attending over a local patch, picks tokens spaced regularly
   across the entire image with stride = window_size. Each of the resulting
   G×G grids (where G = window_size for a square image) sees a sparse but
   GLOBAL set of positions.
   → Captures long-range dependencies cheaply.

Every MaxViT block applies BOTH: local first, then global.
Combined complexity: O(N · P²) instead of O(N²) for full attention.
This gives full global receptive field at linear cost.

Partition helpers (O(1) reshape — no data copy)
-----------------------------------------------
  block_partition(x, block_size):
      [B, H, W, C] → [B * nH * nW, P, P, C]   where P = block_size
      Splits image into a grid of P×P blocks.

  grid_partition(x, grid_size):
      [B, H, W, C] → [B * P * P, G, G, C]      where G = H // grid_size
      Picks every grid_size-th pixel as one "grid window".

Block layout per MaxViT block
------------------------------
  MBConv (depth-wise conv + SE + BN pre-norm)
    ↓
  Block Attention (local window, pre-LN, residual)
    ↓
  Grid Attention (dilated global, pre-LN, residual)

Both attention sub-layers also have a post-attention FFN sublayer.

Stem: Conv(3→dim, 3×3, stride=2) → Conv(dim→dim, 3×3, stride=1)
4 stages: each stage doubles channels and halves spatial resolution.

BackboneOutput
--------------
  pooled_output : [B, dim*8]   — global avg pool of stage 3 output
  feature_maps  : 4 maps, [B, dim*(2^i), H/(2^(i+2)), W/(2^(i+2))]
  patch_tokens  : None
"""

from hale_core.nn.vision.hybrid.maxvit._helpers import (
    block_partition,
    block_reverse,
    grid_partition,
    grid_reverse,
)
from hale_core.nn.vision.hybrid.maxvit.attention_layer import AttentionLayer
from hale_core.nn.vision.hybrid.maxvit.drop_path import DropPath
from hale_core.nn.vision.hybrid.maxvit.max_vi_t_backbone import MaxViTBackbone
from hale_core.nn.vision.hybrid.maxvit.max_vi_t_base import MaxViTBase
from hale_core.nn.vision.hybrid.maxvit.max_vi_t_block import MaxViTBlock
from hale_core.nn.vision.hybrid.maxvit.max_vi_t_large import MaxViTLarge
from hale_core.nn.vision.hybrid.maxvit.max_vi_t_small import MaxViTSmall
from hale_core.nn.vision.hybrid.maxvit.max_vi_t_stage import MaxViTStage
from hale_core.nn.vision.hybrid.maxvit.max_vi_t_tiny import MaxViTTiny
from hale_core.nn.vision.hybrid.maxvit.max_vi_tx_large import MaxViTXLarge
from hale_core.nn.vision.hybrid.maxvit.mb_conv import MBConv
from hale_core.nn.vision.hybrid.maxvit.se import SE
from hale_core.nn.vision.hybrid.maxvit.window_attention import WindowAttention

__all__ = [
    "AttentionLayer",
    "DropPath",
    "MBConv",
    "MaxViTBackbone",
    "MaxViTBase",
    "MaxViTBlock",
    "MaxViTLarge",
    "MaxViTSmall",
    "MaxViTStage",
    "MaxViTTiny",
    "MaxViTXLarge",
    "SE",
    "WindowAttention",
    "block_partition",
    "block_reverse",
    "grid_partition",
    "grid_reverse",
]
