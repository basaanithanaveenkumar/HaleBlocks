"""
Vision Transformer (ViT) Backbone
===================================

Original paper: "An Image is Worth 16x16 Words: Transformers for Image Recognition
at Scale" (Dosovitskiy et al., 2020) — https://arxiv.org/abs/2010.11929

Architecture Overview
---------------------
ViT applies a standard Transformer encoder directly to sequences of flattened 2D
image patches, with minimal inductive bias.  The key insight is that patch sequences
can be treated exactly like word token sequences in NLP.

Pipeline:
    Image [B, 3, H, W]
       ↓  PatchEmbed (Conv2d stride=patch_size)
    Patches [B, N, D]
       ↓  prepend CLS token  +  add positional embedding
    Sequence [B, N+1, D]
       ↓  N×ViTBlock (LayerNorm → MHA → LayerNorm → MLP, pre-norm)
    [B, N+1, D]
       ↓  final LayerNorm
    pooled_output = x[:, 0]  (CLS token)
    patch_tokens  = x[:, 1:] (patch tokens, reshaped to [B, H/p, W/p, D])

Design choices carried forward
--------------------------------
- Pre-norm (LN before Q/K/V), NOT post-norm — more stable for large models
- Single combined in_proj (3×D → 3D), then split, for cache-friendly GEMM layout
- GELU activation in MLP (not ReLU)
- Learned 1D absolute positional embeddings (one vector per patch + CLS)
- Drop-path NOT included here — added in DeiT/DINOv2; keep base ViT minimal
- Dropout in attention (attn_drop) and in MLP (mlp_drop)

Variants registered
-------------------
  vit_tiny   — ViT-Ti/16:   192-dim, 3 heads, 12 layers
  vit_small  — ViT-S/16:    384-dim, 6 heads, 12 layers
  vit_base   — ViT-B/16:    768-dim, 12 heads, 12 layers
  vit_large  — ViT-L/16:   1024-dim, 16 heads, 24 layers
  vit_huge   — ViT-H/14:   1280-dim, 16 heads, 32 layers
  vit_22b    — ViT-22B/14: 6144-dim, 48 heads, 48 layers  (Zhai et al. 2022)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.registry import register_vision_backbone


# ── Initialisation helper ─────────────────────────────────────────────────────

def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    """Truncated normal init at ±2σ — standard for ViT weight init."""
    with torch.no_grad():
        # Box–Muller via normal + clamp, then rescale to correct distribution
        nn.init.trunc_normal_(tensor, mean=0.0, std=std, a=-2 * std, b=2 * std)
    return tensor


# ── Patch embedding ───────────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """Convert an image into non-overlapping patch embeddings.

    Uses a single Conv2d with kernel_size=stride=patch_size, which is
    mathematically equivalent to splitting the image into patches and applying
    a shared linear projection to each, but is faster due to BLAS layout.

    Args:
        img_size:   Assumed square input size (used to pre-compute N_patches).
        patch_size: Side length of each square patch (e.g. 16 or 14).
        in_chans:   Input channels (3 for RGB).
        embed_dim:  Projected dimension D.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size          # e.g. 14 for 224/16
        self.num_patches = self.grid_size ** 2

        # A single strided convolution performs the patch extraction + linear projection
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        _trunc_normal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W] → [B, D, H/p, W/p] → [B, N, D]
        x = self.proj(x)                                 # [B, D, G, G]
        x = x.flatten(2).transpose(1, 2)                # [B, N, D]
        return x


# ── Attention ─────────────────────────────────────────────────────────────────

class ViTAttention(nn.Module):
    """Multi-head self-attention for Vision Transformer.

    Implements the standard scaled dot-product attention:
        Attn(Q,K,V) = softmax(QK^T / √d_head) · V

    Uses a single in_proj weight matrix [D, 3D] for combined Q/K/V — this is
    cache-friendly and matches the original Transformer paper's formulation.

    Args:
        dim:       Input/output dimensionality D.
        num_heads: Number of attention heads H. dim must be divisible by H.
        attn_drop: Dropout probability on attention weights (default 0.0).
        proj_drop: Dropout probability on the output projection (default 0.0).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} must be divisible by num_heads {num_heads}"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5               # 1/√d_head for numerical stability

        # Combined Q, K, V projection — one matrix instead of three for efficiency
        self.in_proj = nn.Linear(dim, dim * 3, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)
        self.attn_drop = nn.Dropout(attn_drop)
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

        # Project and split into Q, K, V heads
        qkv = self.in_proj(x)                            # [B, N, 3D]
        qkv = qkv.reshape(B, N, 3, H, self.head_dim)    # [B, N, 3, H, d]
        qkv = qkv.permute(2, 0, 3, 1, 4)                # [3, B, H, N, d]
        q, k, v = qkv.unbind(0)                          # each [B, H, N, d]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) * self.scale    # [B, H, N, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum of values
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)  # [B, N, D]
        x = self.proj_drop(self.out_proj(x))

        return x, (attn if return_attn else None)


# ── MLP block ─────────────────────────────────────────────────────────────────

class ViTMLP(nn.Module):
    """Two-layer GELU MLP, expand ratio 4× (standard ViT FFN).

    Structure: Linear(D → 4D) → GELU → Dropout → Linear(4D → D) → Dropout
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop2 = nn.Dropout(drop)

        _trunc_normal_(self.fc1.weight)
        _trunc_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.act(self.fc1(x)))
        return self.drop2(self.fc2(x))


# ── Transformer block ─────────────────────────────────────────────────────────

class ViTBlock(nn.Module):
    """Single ViT encoder block: pre-norm → attention → pre-norm → MLP.

    Pre-norm (LN before the residual branch) is more stable than post-norm,
    especially for large models and long sequences.

    Residual formulation:
        x = x + Attention(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        mlp_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ViTAttention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=mlp_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = ViTMLP(dim, mlp_ratio=mlp_ratio, drop=mlp_drop)

    def forward(
        self, x: torch.Tensor, return_attn: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_weights = self.attn(self.norm1(x), return_attn=return_attn)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


# ── Full ViT backbone ─────────────────────────────────────────────────────────

class ViTBackbone(BaseVisionBackbone):
    """Standard Vision Transformer backbone.

    Returns a BackboneOutput where:
      - pooled_output:  CLS token embedding [B, embed_dim]
      - patch_tokens:   All patch token embeddings [B, N, embed_dim]
      - feature_maps:   Patch tokens reshaped to spatial grid [B, D, H/p, W/p]
                        Single-scale (ViT has no multi-scale FPN hierarchy)

    Args:
        embed_dim:    Token dimension D.
        depth:        Number of transformer blocks.
        num_heads:    Attention heads per block.
        patch_size:   Side length of each patch (16 or 14).
        mlp_ratio:    FFN hidden dim multiplier (default 4.0).
        attn_drop:    Dropout on attention weights.
        drop_rate:    Dropout on patch embeddings and MLP outputs.
        return_all_attn: If True, cache attention maps from all blocks.
        img_size, in_chans, out_indices: inherited from BaseVisionBackbone.
    """

    arch_name: str = "vit"

    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        patch_size: int = 16,
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
        self.feature_dims = [embed_dim]  # single-scale ViT

        # Patch projection
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        # CLS token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ViTBlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, attn_drop=attn_drop, mlp_drop=drop_rate)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        _trunc_normal_(self.cls_token)
        _trunc_normal_(self.pos_embed)

    def freeze_up_to_stage(self, stage: int) -> None:
        """Freeze the first `stage` transformer blocks (0-indexed).

        Stage mapping:
          stage=0 → freeze patch_embed, cls_token, pos_embed, blocks[0]
          stage=k → additionally freeze blocks[1..k]
        """
        # Freeze embeddings
        self.patch_embed.requires_grad_(False)
        self.cls_token.requires_grad_(False)
        self.pos_embed.requires_grad_(False)
        # Freeze first (stage+1) blocks
        for i in range(min(stage + 1, self.depth)):
            self.blocks[i].requires_grad_(False)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        B = x.shape[0]

        # 1. Patch embedding
        x = self.patch_embed(x)                          # [B, N, D]

        # 2. Prepend CLS token and add positional embedding
        cls = self.cls_token.expand(B, -1, -1)           # [B, 1, D]
        x = torch.cat([cls, x], dim=1)                   # [B, N+1, D]
        x = self.pos_drop(x + self.pos_embed)

        # 3. Transformer blocks
        attn_maps: list[torch.Tensor] = []
        for blk in self.blocks:
            x, attn = blk(x, return_attn=self.return_all_attn)
            if self.return_all_attn and attn is not None:
                attn_maps.append(attn)

        x = self.norm(x)                                  # [B, N+1, D]

        # 4. Unpack outputs
        pooled = x[:, 0]                                  # [B, D]  — CLS token
        patches = x[:, 1:]                                # [B, N, D]

        # Reshape to spatial grid for dense tasks
        G = self.patch_embed.grid_size
        feat_map = patches.transpose(1, 2).reshape(B, self.embed_dim, G, G)  # [B, D, G, G]

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=patches,
            feature_maps=[feat_map],
            attentions=attn_maps if self.return_all_attn else None,
        )


# ── Variant factories ─────────────────────────────────────────────────────────
# Each variant is a thin subclass that sets arch_name + default constructor kwargs.
# The @register_vision_backbone decorator adds it to the global backbone registry.

@register_vision_backbone("vit_tiny")
class ViTTiny(ViTBackbone):
    """ViT-Ti/16 — 5.7M params. Fastest ViT; useful for ablations."""
    arch_name = "vit_tiny"
    feature_dims = [192]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 192)
        kwargs.setdefault("depth", 12)
        kwargs.setdefault("num_heads", 3)
        kwargs.setdefault("patch_size", 16)
        super().__init__(**kwargs)


@register_vision_backbone("vit_small")
class ViTSmall(ViTBackbone):
    """ViT-S/16 — 22M params. Good accuracy/speed trade-off."""
    arch_name = "vit_small"
    feature_dims = [384]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 384)
        kwargs.setdefault("depth", 12)
        kwargs.setdefault("num_heads", 6)
        kwargs.setdefault("patch_size", 16)
        super().__init__(**kwargs)


@register_vision_backbone("vit_base")
class ViTBase(ViTBackbone):
    """ViT-B/16 — 86M params. The standard reference ViT."""
    arch_name = "vit_base"
    feature_dims = [768]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 768)
        kwargs.setdefault("depth", 12)
        kwargs.setdefault("num_heads", 12)
        kwargs.setdefault("patch_size", 16)
        super().__init__(**kwargs)


@register_vision_backbone("vit_large")
class ViTLarge(ViTBackbone):
    """ViT-L/16 — 307M params. High-capacity model for demanding tasks."""
    arch_name = "vit_large"
    feature_dims = [1024]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 1024)
        kwargs.setdefault("depth", 24)
        kwargs.setdefault("num_heads", 16)
        kwargs.setdefault("patch_size", 16)
        super().__init__(**kwargs)


@register_vision_backbone("vit_huge")
class ViTHuge(ViTBackbone):
    """ViT-H/14 — 632M params. Maximum standard ViT size."""
    arch_name = "vit_huge"
    feature_dims = [1280]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 1280)
        kwargs.setdefault("depth", 32)
        kwargs.setdefault("num_heads", 16)
        kwargs.setdefault("patch_size", 14)
        super().__init__(**kwargs)


@register_vision_backbone("vit_22b")
class ViT22B(ViTBackbone):
    """ViT-22B/14 — 22B params (Zhai et al., 2022).

    At this scale the model is expected to be sharded across devices.
    The class provides the architecture; distributed loading is caller's responsibility.

    Reference: "Scaling Vision Transformers to 22 Billion Parameters" (Zhai 2022)
               https://arxiv.org/abs/2302.05442
    """
    arch_name = "vit_22b"
    feature_dims = [6144]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 6144)
        kwargs.setdefault("depth", 48)
        kwargs.setdefault("num_heads", 48)
        kwargs.setdefault("patch_size", 14)
        super().__init__(**kwargs)
