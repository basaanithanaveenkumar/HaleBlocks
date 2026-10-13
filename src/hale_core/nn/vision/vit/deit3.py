"""
DeiT III Vision Transformer Backbone
======================================

Paper: "DeiT III: Revenge of the ViT" (Touvron et al., 2022)
       https://arxiv.org/abs/2204.07118

Architecture Overview
---------------------
DeiT III demonstrates that a vanilla ViT trained with the right data augmentation
and regularisation recipe (3-Augmentation, or 3-Reg) matches or beats more complex
architectures.  The architectural changes over baseline ViT are minimal:

1. LayerScale (Touvron et al., 2021 — CaiT)
   Same as DINOv2: per-channel learnable scale γ on each residual branch.
   Initial value: 1e-6 for large models (smaller than DINOv2's 1e-5 to handle
   more aggressive augmentation at the start of training).
   γ is a free parameter — at convergence it may be near 1.0 (full residual)
   or remain small (dampened contribution).

2. No CLS token — mean-pooling
   Unlike baseline ViT which uses a CLS token, DeiT III pools all patch tokens:
       pooled = mean(patch_tokens, dim=1)
   This removes the asymmetry between CLS and patch tokens, and all patches
   contribute equally to the global representation.  This simplifies transfer
   learning: any patch can be used for dense prediction without the CLS token
   occupying a "special" position.

3. Absolute learned positional embeddings (same as ViT)
   DeiT III does NOT use RoPE — it keeps the simple learned absolute embeddings.
   The 3-Reg augmentation (Random Resize Crop, RandAugment, Random Erasing) is
   the key to the strong performance, not architecture changes.

3-Augmentation training recipe (not implemented here, documented for reference)
  - RandomResizedCrop(224)
  - RandAugment(n=2, m=9) — two magnitude-9 transformations per image
  - RandomErasing(prob=0.25, scale=(0.02, 0.33))
  - Mixup(alpha=0.8) + CutMix(alpha=1.0)
  - Label smoothing (ε=0.1)
  - Repeated augmentation (same image augmented twice per batch for diversity)
  This recipe is applied in the training pipeline, not the backbone.

Variants registered
-------------------
  deit3_small — 384-dim, 6 heads,  12 layers, patch=16
  deit3_base  — 768-dim, 12 heads, 12 layers, patch=16
  deit3_large — 1024-dim, 16 heads, 24 layers, patch=16
  deit3_huge  — 1280-dim, 16 heads, 32 layers, patch=14
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.registry import register_vision_backbone
from hale_core.nn.vision.vit.vit import PatchEmbed, ViTMLP, _trunc_normal_
from hale_core.nn.vision.vit.dino import LayerScale


# ── Attention (identical to DINOv2 but without register token logic) ──────────

class DeiT3Attention(nn.Module):
    """Standard multi-head self-attention for DeiT III.

    Uses PyTorch 2.0 fused kernel (F.scaled_dot_product_attention) when available.
    No register tokens, no LayerScale inside the attention — LayerScale is on the block.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.attn_drop_p = attn_drop
        self._use_flash = hasattr(F, "scaled_dot_product_attention")

        self.in_proj = nn.Linear(dim, dim * 3, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

        _trunc_normal_(self.in_proj.weight)
        _trunc_normal_(self.out_proj.weight)
        nn.init.zeros_(self.in_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = x.shape
        H = self.num_heads

        qkv = self.in_proj(x).reshape(B, N, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self._use_flash and not return_attn:
            drop_p = self.attn_drop_p if self.training else 0.0
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)
            attn_weights = None
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            if self.training and self.attn_drop_p > 0:
                attn = F.dropout(attn, p=self.attn_drop_p)
            attn_weights = attn if return_attn else None
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.out_proj(x)), attn_weights


# ── DeiT III block ────────────────────────────────────────────────────────────

class DeiT3Block(nn.Module):
    """DeiT III transformer block: pre-norm + LayerScale, GELU MLP.

    Key: LayerScale init is 1e-6 (smaller than DINOv2's 1e-5) to accommodate
    the more aggressive 3-Augmentation training recipe that perturbs gradients
    more strongly at the start.

    Residual:
        x = x + LayerScale_attn(Attention(LN(x)))
        x = x + LayerScale_mlp(MLP(LN(x)))
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        mlp_drop: float = 0.0,
        ls_init: float = 1e-6,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DeiT3Attention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=mlp_drop)
        self.ls1 = LayerScale(dim, init_value=ls_init)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = ViTMLP(dim, mlp_ratio=mlp_ratio, drop=mlp_drop)
        self.ls2 = LayerScale(dim, init_value=ls_init)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_weights = self.attn(self.norm1(x), return_attn=return_attn)
        x = x + self.ls1(attn_out)
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x, attn_weights


# ── Full DeiT III backbone ────────────────────────────────────────────────────

class DeiT3Backbone(BaseVisionBackbone):
    """DeiT III backbone: ViT with LayerScale and mean-pool (no CLS token).

    Output:
      - pooled_output:  mean(patch_tokens, dim=1) [B, D]
      - patch_tokens:   all patch tokens [B, N, D]
      - feature_maps:   patch tokens as spatial grid [B, D, H/p, W/p]

    Args:
        embed_dim:     Token dimension D.
        depth:         Number of transformer blocks.
        num_heads:     Attention heads per block.
        patch_size:    Patch side length.
        ls_init:       LayerScale initial value (default 1e-6).
        mlp_ratio:     FFN expansion ratio (default 4.0).
        return_all_attn: Cache attention maps from all blocks.
    """

    arch_name: str = "deit3"

    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        patch_size: int = 16,
        ls_init: float = 1e-6,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        drop_rate: float = 0.0,
        return_all_attn: bool = False,
        out_indices: list[int] | None = None,
    ) -> None:
        super().__init__(img_size=img_size, in_chans=in_chans, drop_rate=drop_rate, out_indices=out_indices)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depth = depth
        self.return_all_attn = return_all_attn
        self.feature_dims = [embed_dim]

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # Absolute learned positional embedding (no CLS token, so just N patches)
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.ModuleList([
            DeiT3Block(embed_dim, num_heads, mlp_ratio=mlp_ratio, attn_drop=attn_drop, mlp_drop=drop_rate, ls_init=ls_init)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        _trunc_normal_(self.pos_embed)

    def freeze_up_to_stage(self, stage: int) -> None:
        """Freeze patch embed, pos embed, and the first `stage` blocks."""
        self.patch_embed.requires_grad_(False)
        self.pos_embed.requires_grad_(False)
        for i in range(min(stage + 1, self.depth)):
            self.blocks[i].requires_grad_(False)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        B = x.shape[0]

        # 1. Patch embedding + positional encoding
        x = self.patch_embed(x)                         # [B, N, D]
        x = self.pos_drop(x + self.pos_embed)

        # 2. Transformer blocks
        attn_maps: list[torch.Tensor] = []
        for blk in self.blocks:
            x, attn = blk(x, return_attn=self.return_all_attn)
            if self.return_all_attn and attn is not None:
                attn_maps.append(attn)

        x = self.norm(x)                                # [B, N, D]

        # 3. Mean pooling over all patches (no special CLS token)
        pooled = x.mean(dim=1)                          # [B, D]

        # Spatial reshape for dense prediction heads
        G = self.patch_embed.grid_size
        feat_map = x.transpose(1, 2).reshape(B, self.embed_dim, G, G)  # [B, D, G, G]

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=x,
            feature_maps=[feat_map],
            attentions=attn_maps if self.return_all_attn else None,
        )


# ── Variant factories ─────────────────────────────────────────────────────────

@register_vision_backbone("deit3_small")
class DeiT3Small(DeiT3Backbone):
    """DeiT III-S/16 — 22M params."""
    arch_name = "deit3_small"
    feature_dims = [384]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 384)
        kwargs.setdefault("depth", 12)
        kwargs.setdefault("num_heads", 6)
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("ls_init", 1e-6)
        super().__init__(**kwargs)


@register_vision_backbone("deit3_base")
class DeiT3Base(DeiT3Backbone):
    """DeiT III-B/16 — 86M params."""
    arch_name = "deit3_base"
    feature_dims = [768]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 768)
        kwargs.setdefault("depth", 12)
        kwargs.setdefault("num_heads", 12)
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("ls_init", 1e-6)
        super().__init__(**kwargs)


@register_vision_backbone("deit3_large")
class DeiT3Large(DeiT3Backbone):
    """DeiT III-L/16 — 307M params."""
    arch_name = "deit3_large"
    feature_dims = [1024]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 1024)
        kwargs.setdefault("depth", 24)
        kwargs.setdefault("num_heads", 16)
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("ls_init", 1e-6)
        super().__init__(**kwargs)


@register_vision_backbone("deit3_huge")
class DeiT3Huge(DeiT3Backbone):
    """DeiT III-H/14 — 632M params.  The flagship DeiT III model."""
    arch_name = "deit3_huge"
    feature_dims = [1280]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 1280)
        kwargs.setdefault("depth", 32)
        kwargs.setdefault("num_heads", 16)
        kwargs.setdefault("patch_size", 14)
        kwargs.setdefault("ls_init", 1e-6)
        super().__init__(**kwargs)
