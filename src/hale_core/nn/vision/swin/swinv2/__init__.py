"""
SwinV2 — Swin Transformer V2 (Liu et al., 2022)
================================================

Reference: "Swin Transformer V2: Scaling Up Capacity and Resolution"
           https://arxiv.org/abs/2111.09883

Architecture overview
---------------------
SwinV2 is a hierarchical vision transformer that operates on shifted windows,
making it efficient for high-resolution inputs. It introduces three key upgrades
over SwinV1 to enable scaling to billions of parameters and gigapixel images:

1. **Residual Post-Norm** (instead of pre-norm):
   LayerNorm is applied AFTER the residual addition. This stabilises gradient
   flow in very deep networks and allows training with larger models.

2. **Log-spaced Continuous Relative Position Bias (Log-CPB)**:
   Instead of a learnable table indexed by integer offsets (which is fixed to
   the training window size), Log-CPB uses a small MLP that maps log-compressed
   (dx, dy) differences to a bias scalar. This allows fine-tuning at window
   sizes different from pretraining without discarding learned position info.
       log_cpb(dx, dy) = MLP(sign(dx)·log(1+|dx|), sign(dy)·log(1+|dy|))

3. **Scaled Cosine Attention**:
   Instead of dot-product attention, uses:
       attn(Q, K) = cos(Q, K) / tau + B
   where tau is a learnable log-scale per-head temperature (init=log(10)),
   preventing attention logit explosion in very deep models.

Window mechanics
----------------
- Input [B, H, W, C] is partitioned into (H//win) × (W//win) windows of size win×win.
- Self-attention runs independently inside each window (local receptive field).
- For SHIFTED windows, the feature map is cyclically shifted by (win//2, win//2)
  before partitioning, then un-shifted after. An attention mask prevents tokens
  in different "regions" from attending to each other across the shift boundary.

Multi-scale hierarchy
---------------------
Stage 0: patch embed (4×4 stride) → 2 blocks,         output H/4  × W/4  × C
Stage 1: patch merge (2×2 stride) → 2..42 blocks,     output H/8  × W/8  × 2C
Stage 2: patch merge              → N blocks,          output H/16 × W/16 × 4C
Stage 3: patch merge              → 2..4 blocks,       output H/32 × W/32 × 8C

BackboneOutput
--------------
  pooled_output  : [B, 8C]           — global average of stage-3 features
  feature_maps   : [[B,C,H/4,W/4], [B,2C,H/8,W/8], [B,4C,H/16,W/16], [B,8C,H/32,W/32]]
  patch_tokens   : None              — not meaningful for SwinV2 (spatial, not seq)
"""

from hale_core.nn.vision.swin.swinv2._helpers import window_partition, window_reverse
from hale_core.nn.vision.swin.swinv2.drop_path import DropPath
from hale_core.nn.vision.swin.swinv2.log_continuous_position_bias import LogContinuousPositionBias
from hale_core.nn.vision.swin.swinv2.patch_embed import PatchEmbed
from hale_core.nn.vision.swin.swinv2.patch_merging import PatchMerging
from hale_core.nn.vision.swin.swinv2.swin_v2_backbone import SwinV2Backbone
from hale_core.nn.vision.swin.swinv2.swin_v2_base import SwinV2Base
from hale_core.nn.vision.swin.swinv2.swin_v2_block import SwinV2Block
from hale_core.nn.vision.swin.swinv2.swin_v2_giant import SwinV2Giant
from hale_core.nn.vision.swin.swinv2.swin_v2_large import SwinV2Large
from hale_core.nn.vision.swin.swinv2.swin_v2_small import SwinV2Small
from hale_core.nn.vision.swin.swinv2.swin_v2_stage import SwinV2Stage
from hale_core.nn.vision.swin.swinv2.swin_v2_tiny import SwinV2Tiny
from hale_core.nn.vision.swin.swinv2.swin_v2_window_attention import SwinV2WindowAttention

__all__ = [
    "DropPath",
    "LogContinuousPositionBias",
    "PatchEmbed",
    "PatchMerging",
    "SwinV2Backbone",
    "SwinV2Base",
    "SwinV2Block",
    "SwinV2Giant",
    "SwinV2Large",
    "SwinV2Small",
    "SwinV2Stage",
    "SwinV2Tiny",
    "SwinV2WindowAttention",
    "window_partition",
    "window_reverse",
]
