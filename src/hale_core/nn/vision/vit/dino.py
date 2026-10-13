"""
DINOv2 Vision Transformer Backbone
=====================================

Paper: "DINOv2: Learning Robust Visual Features without Supervision"
       (Oquab et al., 2023) — https://arxiv.org/abs/2304.07193

Architecture Overview
---------------------
DINOv2 builds on ViT with three key improvements for self-supervised training:

1. Register Tokens
   Large ViTs develop high-norm artifact tokens in background regions.  DINOv2
   introduces n_register learnable tokens inserted AFTER the CLS token.  Registers
   absorb global image statistics (background, repetitive textures), freeing patch
   tokens to carry semantically meaningful local features.  At inference they are
   discarded — only CLS and patch tokens form the output.

2. LayerScale (Caron et al., 2021 — CaiT)
   A per-channel learnable scalar γ initialised at a small value (1e-5 for large
   models) multiplied onto each residual output:
       x = x + γ · Attention(LN(x))
   This prevents deep networks from exploding gradients during early training by
   making residuals near-zero initially, then letting γ grow naturally.

3. Flash Attention (F.scaled_dot_product_attention)
   Uses PyTorch 2.0's fused kernel when available.  Falls back to manual attention
   on older versions.  No numerical difference — the fused kernel is just faster.

Variants registered
-------------------
  dino_vit_small  — 384-dim, 6 heads,  12 layers, patch=14, 4 registers
  dino_vit_base   — 768-dim, 12 heads, 12 layers, patch=14, 4 registers
  dino_vit_large  — 1024-dim, 16 heads, 24 layers, patch=14, 4 registers
  dino_vit_giant  — 1536-dim, 24 heads, 40 layers, patch=14, 4 registers  ← DINOv2-g
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.registry import register_vision_backbone

# Reuse PatchEmbed and MLP from sibling vit module (avoid duplication)
from hale_core.nn.vision.vit.vit import PatchEmbed, ViTMLP, _trunc_normal_


# ── LayerScale ────────────────────────────────────────────────────────────────

class LayerScale(nn.Module):
    """Per-channel learned scale applied to a residual branch.

    Initialised at `init_value` (≈1e-5 for large models) so residuals start
    near zero.  The scale is learned and grows during training, allowing the
    network to gradually increase the residual contribution.

    Reference: "Going Deeper with Image Transformers" (Touvron et al., 2021)
               https://arxiv.org/abs/2103.17239
    """

    def __init__(self, dim: int, init_value: float = 1e-5) -> None:
        super().__init__()
        # Scalar per output channel; starts small to dampen residuals initially
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


# ── Attention with flash-attn fallback ───────────────────────────────────────

class DINOAttention(nn.Module):
    """Multi-head self-attention with optional PyTorch 2.0 fused kernel.

    When torch.nn.functional.scaled_dot_product_attention is available (PyTorch
    ≥ 2.0), uses the fused Flash Attention kernel for ~2× speedup on long
    sequences.  Falls back to manual attention on older versions.

    Args:
        dim:       Token dimension D.
        num_heads: Number of attention heads H.
        attn_drop: Dropout probability on attention weights (ignored for flash attn).
        proj_drop: Dropout probability on output projection.
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

        self.in_proj = nn.Linear(dim, dim * 3, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.proj_drop = nn.Dropout(proj_drop)

        # Check once at init whether fused kernel is available
        self._use_flash = hasattr(F, "scaled_dot_product_attention")

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
        q, k, v = qkv.unbind(0)                          # each [B, H, N, d]

        if self._use_flash and not return_attn:
            # Fused Flash Attention — no explicit attention matrix materialised
            # dropout_p only applies during training; set 0.0 at eval
            drop_p = self.attn_drop_p if self.training else 0.0
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)
            attn_weights = None
        else:
            # Manual attention (needed when return_attn=True for visualisation)
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = F.softmax(attn, dim=-1)
            if self.training and self.attn_drop_p > 0:
                attn = F.dropout(attn, p=self.attn_drop_p)
            attn_weights = attn if return_attn else None
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.out_proj(x)), attn_weights


# ── Transformer block with LayerScale ────────────────────────────────────────

class DINOBlock(nn.Module):
    """DINOv2 transformer block: pre-norm + LayerScale on both branches.

    Residual:
        x = x + LayerScale(Attention(LN(x)))
        x = x + LayerScale(MLP(LN(x)))
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        mlp_drop: float = 0.0,
        ls_init: float = 1e-5,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = DINOAttention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=mlp_drop)
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


# ── Full DINOv2 backbone ──────────────────────────────────────────────────────

class DINOv2Backbone(BaseVisionBackbone):
    """DINOv2 ViT backbone with register tokens and LayerScale.

    Output:
      - pooled_output:  CLS token embedding [B, D]
      - patch_tokens:   Patch tokens (register tokens discarded) [B, N, D]
      - feature_maps:   Patch tokens as spatial grid [B, D, H/p, W/p]

    Args:
        embed_dim:     Token dimension D.
        depth:         Number of transformer blocks.
        num_heads:     Attention heads.
        patch_size:    Patch side length (default 14 for DINOv2).
        n_register:    Number of register tokens (default 4).
        use_cls:       Whether to use a CLS token (True by default).
                       If False, mean-pools all patch tokens as pooled_output.
        ls_init:       LayerScale initial value (1e-5 for large, 1.0 disables).
        mlp_ratio:     FFN expansion ratio.
        return_all_attn: Cache attention maps from all blocks.
    """

    arch_name: str = "dino"

    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        patch_size: int = 14,
        n_register: int = 4,
        use_cls: bool = True,
        ls_init: float = 1e-5,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        drop_rate: float = 0.0,
        return_all_attn: bool = False,
        out_indices: list[int] | None = None,
    ) -> None:
        super().__init__(img_size=img_size, in_chans=in_chans, drop_rate=drop_rate, out_indices=out_indices)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.n_register = n_register
        self.use_cls = use_cls
        self.depth = depth
        self.return_all_attn = return_all_attn
        self.feature_dims = [embed_dim]

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token and positional embedding (pos emb covers CLS + patches only, not registers)
        if use_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        # Register tokens — extra learnable tokens inserted after CLS
        self.register_tokens = nn.Parameter(torch.zeros(1, n_register, embed_dim))

        # Positional embedding for CLS + patches (registers share no positional info — design choice)
        n_cls = 1 if use_cls else 0
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + n_cls, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        self.blocks = nn.ModuleList([
            DINOBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, attn_drop=attn_drop, mlp_drop=drop_rate, ls_init=ls_init)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        if self.cls_token is not None:
            _trunc_normal_(self.cls_token)
        _trunc_normal_(self.register_tokens)
        _trunc_normal_(self.pos_embed)

    def freeze_up_to_stage(self, stage: int) -> None:
        """Freeze embeddings and the first `stage` blocks."""
        self.patch_embed.requires_grad_(False)
        if self.cls_token is not None:
            self.cls_token.requires_grad_(False)
        self.register_tokens.requires_grad_(False)
        self.pos_embed.requires_grad_(False)
        for i in range(min(stage + 1, self.depth)):
            self.blocks[i].requires_grad_(False)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        B = x.shape[0]

        # 1. Patch embedding
        patches = self.patch_embed(x)                    # [B, N, D]

        # 2. Build token sequence: [CLS, patches] then add positional embedding
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)       # [B, 1, D]
            tokens = torch.cat([cls, patches], dim=1)    # [B, 1+N, D]
        else:
            tokens = patches                             # [B, N, D]

        tokens = self.pos_drop(tokens + self.pos_embed)

        # 3. Insert register tokens AFTER positional embedding (they don't have positions)
        regs = self.register_tokens.expand(B, -1, -1)   # [B, n_reg, D]
        if self.use_cls:
            # Insert registers between CLS and patches: [CLS, reg..., patches]
            tokens = torch.cat([tokens[:, :1], regs, tokens[:, 1:]], dim=1)
        else:
            # No CLS: [reg..., patches]
            tokens = torch.cat([regs, tokens], dim=1)

        # 4. Transformer blocks
        attn_maps: list[torch.Tensor] = []
        for blk in self.blocks:
            tokens, attn = blk(tokens, return_attn=self.return_all_attn)
            if self.return_all_attn and attn is not None:
                attn_maps.append(attn)

        tokens = self.norm(tokens)

        # 5. Extract outputs — discard register tokens
        if self.use_cls:
            n_prefix = 1 + self.n_register              # CLS + registers
            pooled = tokens[:, 0]                       # CLS token [B, D]
            patches_out = tokens[:, n_prefix:]          # patch tokens [B, N, D]
        else:
            pooled = tokens[:, self.n_register:].mean(dim=1)  # mean pool over patches
            patches_out = tokens[:, self.n_register:]

        G = self.patch_embed.grid_size
        feat_map = patches_out.transpose(1, 2).reshape(B, self.embed_dim, G, G)

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=patches_out,
            feature_maps=[feat_map],
            attentions=attn_maps if self.return_all_attn else None,
        )


# ── Variant factories ─────────────────────────────────────────────────────────

@register_vision_backbone("dino_vit_small")
class DINOViTSmall(DINOv2Backbone):
    """DINOv2 ViT-S/14 — 22M params."""
    arch_name = "dino_vit_small"
    feature_dims = [384]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 384)
        kwargs.setdefault("depth", 12)
        kwargs.setdefault("num_heads", 6)
        kwargs.setdefault("patch_size", 14)
        kwargs.setdefault("n_register", 4)
        kwargs.setdefault("ls_init", 1e-5)
        super().__init__(**kwargs)


@register_vision_backbone("dino_vit_base")
class DINOViTBase(DINOv2Backbone):
    """DINOv2 ViT-B/14 — 86M params."""
    arch_name = "dino_vit_base"
    feature_dims = [768]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 768)
        kwargs.setdefault("depth", 12)
        kwargs.setdefault("num_heads", 12)
        kwargs.setdefault("patch_size", 14)
        kwargs.setdefault("n_register", 4)
        kwargs.setdefault("ls_init", 1e-5)
        super().__init__(**kwargs)


@register_vision_backbone("dino_vit_large")
class DINOViTLarge(DINOv2Backbone):
    """DINOv2 ViT-L/14 — 307M params."""
    arch_name = "dino_vit_large"
    feature_dims = [1024]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 1024)
        kwargs.setdefault("depth", 24)
        kwargs.setdefault("num_heads", 16)
        kwargs.setdefault("patch_size", 14)
        kwargs.setdefault("n_register", 4)
        kwargs.setdefault("ls_init", 1e-5)
        super().__init__(**kwargs)


@register_vision_backbone("dino_vit_giant")
class DINOViTGiant(DINOv2Backbone):
    """DINOv2 ViT-G/14 — 1.1B params.  The flagship DINOv2 model."""
    arch_name = "dino_vit_giant"
    feature_dims = [1536]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 1536)
        kwargs.setdefault("depth", 40)
        kwargs.setdefault("num_heads", 24)
        kwargs.setdefault("patch_size", 14)
        kwargs.setdefault("n_register", 4)
        kwargs.setdefault("ls_init", 1e-5)
        super().__init__(**kwargs)
