"""
EVA-02 Vision Transformer Backbone
=====================================

Paper: "EVA-02: A Visual Representation Powerhouse Reconstructed from EVA"
       (Fang et al., 2023) — https://arxiv.org/abs/2303.11331

Architecture Overview
---------------------
EVA-02 introduces three key modifications over plain ViT that together achieve
state-of-the-art visual representations:

1. Rotary Position Embedding (2D RoPE)
   Instead of learned absolute positional embeddings added to all tokens, RoPE
   encodes positions by rotating Q and K vectors with position-dependent angles.

   For a 1D sequence, RoPE rotates consecutive pairs (x_{2i}, x_{2i+1}) by θ:
       x' = x·cos(θ) + rotate_half(x)·sin(θ)
   where rotate_half swaps pairs with a sign flip: [a, b, c, d] → [-b, a, -d, c].

   For 2D images, EVA-02 uses separate RoPE for rows and columns:
       - Each head's d/2 dimensions are split into d/4 row + d/4 col dims
       - Row dims rotate by position_y * θ_i, col dims by position_x * θ_i
   This generalises to arbitrary resolutions without interpolation artefacts,
   unlike absolute learned positional embeddings.

2. SwiGLU FFN
   Standard ViT uses GELU(Linear(x)) as the activation gate.  SwiGLU uses:
       SwiGLU(x) = SiLU(W₁x) ⊙ W₂x   (element-wise gate)
   This is a gated linear unit (GLU) variant shown to outperform GELU MLPs at
   the same parameter count (Noam Shazeer, 2020: "GLU Variants Improve Transformer").
   The intermediate dim is scaled by 2/3 × 4 = 8/3 ≈ 2.667× to keep parameter
   count equal to a standard 4× MLP (two matrices instead of one gate).

3. Sub-Layer Normalization (Sub-LN) for Q and K
   Applying LayerNorm to Q and K before computing attention stabilises training
   by preventing attention logit explosion at large scale (large dim, many heads).
   This is independent from the pre-norm applied to the full token sequence.

Output:
   EVA-02 uses mean-pooling over all patch tokens (no CLS token in EVA-02 style)
   as the global pooled_output.

Variants registered
-------------------
  eva_02_small — 384-dim, 6 heads,  12 layers, patch=16
  eva_02_base  — 768-dim, 12 heads, 12 layers, patch=16
  eva_02_large — 1024-dim, 16 heads, 24 layers, patch=14
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.registry import register_vision_backbone
from hale_core.nn.vision.vit.vit import PatchEmbed, _trunc_normal_


# ── 2D Rotary Position Embedding ─────────────────────────────────────────────

def _build_rope_freqs(
    grid_size: int,
    head_dim: int,
    base: float = 10000.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute RoPE sin/cos frequencies for a 2D grid.

    Each head has `head_dim` dimensions.  We use head_dim/2 for row positions
    and head_dim/2 for column positions.

    Returns:
        cos_freq: [N, head_dim]  (N = grid_size²)
        sin_freq: [N, head_dim]
    """
    # Frequency bands — one per pair of dimensions
    half = head_dim // 4                             # row uses half, col uses half
    theta = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))

    # Grid positions
    gy = torch.arange(grid_size, dtype=torch.float32)
    gx = torch.arange(grid_size, dtype=torch.float32)
    # Outer product → N positions
    py = gy.unsqueeze(1).expand(grid_size, grid_size).reshape(-1)  # [N]
    px = gx.unsqueeze(0).expand(grid_size, grid_size).reshape(-1)  # [N]

    # Angles per position per frequency
    ang_y = py.unsqueeze(1) * theta.unsqueeze(0)    # [N, half]
    ang_x = px.unsqueeze(1) * theta.unsqueeze(0)    # [N, half]

    # Concat row and col angles → [N, head_dim/2]
    ang = torch.cat([ang_y, ang_x], dim=1)          # [N, head_dim/2]

    # Duplicate so each pair gets the same angle: [N, head_dim]
    # Pattern: [θ₀, θ₀, θ₁, θ₁, ...] → cos/sin applied to (x₀, x₁) pair
    ang = ang.repeat_interleave(2, dim=1)

    return ang.cos(), ang.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate pairs of consecutive dimensions: [a, b, c, d] → [-b, a, -d, c]."""
    x1, x2 = x[..., ::2], x[..., 1::2]             # split even/odd dims
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def _apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE rotation to Q and K tensors.

    Args:
        q, k:    [B, H, N, d]
        cos, sin: [N, d]  (pre-computed for the sequence length N)

    Returns:
        q', k' with RoPE applied — same shape.
    """
    # Broadcast cos/sin over batch and head dimensions
    cos = cos.unsqueeze(0).unsqueeze(0)              # [1, 1, N, d]
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_rot = q * cos + _rotate_half(q) * sin
    k_rot = k * cos + _rotate_half(k) * sin
    return q_rot, k_rot


# ── SwiGLU FFN ───────────────────────────────────────────────────────────────

class SwiGLUFFN(nn.Module):
    """SwiGLU gated feed-forward network.

    Computes:  SiLU(W₁·x) ⊙ W₂·x → W₃

    The intermediate dim is 8/3 × dim (not 4×) so that the total parameter
    count matches a standard 4× GELU MLP (two projections instead of one).

    Reference: "GLU Variants Improve Transformer" (Shazeer, 2020)
    """

    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        # Scale hidden dim by 2/3 to keep total params equal to GELU MLP at same ratio
        hidden = int(dim * mlp_ratio * 2 / 3)
        # Make divisible by 256 for efficient GEMM
        hidden = (hidden + 255) // 256 * 256

        self.w1 = nn.Linear(dim, hidden, bias=False)   # gate branch (SiLU)
        self.w2 = nn.Linear(dim, hidden, bias=False)   # value branch
        self.w3 = nn.Linear(hidden, dim, bias=False)   # output projection
        self.drop = nn.Dropout(drop)

        _trunc_normal_(self.w1.weight)
        _trunc_normal_(self.w2.weight)
        _trunc_normal_(self.w3.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SiLU gate applied element-wise to the value branch
        gate = F.silu(self.w1(x))
        value = self.w2(x)
        return self.drop(self.w3(gate * value))


# ── EVA-02 Attention with RoPE and Sub-LN ────────────────────────────────────

class EVAAttention(nn.Module):
    """Multi-head self-attention with RoPE and per-QK LayerNorm.

    Sub-LN: Separate LayerNorm on Q and K before computing attention.
    This prevents attention logit explosion observed in very large models.

    RoPE: The pre-computed cos/sin buffers must be passed from the backbone
    (they depend on the image resolution, which is fixed at __init__ time).

    Args:
        dim:       Token dimension D.
        num_heads: Number of attention heads H.
        attn_drop: Attention weight dropout.
        proj_drop: Output projection dropout.
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

        # Sub-LN: LayerNorm on Q and K to stabilise deep network training
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)

        _trunc_normal_(self.in_proj.weight)
        _trunc_normal_(self.out_proj.weight)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = x.shape
        H = self.num_heads

        qkv = self.in_proj(x).reshape(B, N, 3, H, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)                          # each [B, H, N, d]

        # Sub-LN on Q and K (per-head, over head_dim axis)
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply 2D RoPE to Q and K — positions come from the spatial grid
        q, k = _apply_rope(q, k, rope_cos.to(q.dtype), rope_sin.to(k.dtype))

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        if self.training and self.attn_drop_p > 0:
            attn = F.dropout(attn, p=self.attn_drop_p)
        attn_weights = attn if return_attn else None

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.out_proj(x)), attn_weights


# ── EVA-02 Block ──────────────────────────────────────────────────────────────

class EVABlock(nn.Module):
    """EVA-02 transformer block: pre-norm, RoPE attention, SwiGLU FFN.

    RoPE frequencies are passed from the backbone and not stored per-block
    to avoid duplicating large buffers in memory.
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
        self.attn = EVAAttention(dim, num_heads=num_heads, attn_drop=attn_drop, proj_drop=mlp_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = SwiGLUFFN(dim, mlp_ratio=mlp_ratio, drop=mlp_drop)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: torch.Tensor,
        rope_sin: torch.Tensor,
        return_attn: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        attn_out, attn_weights = self.attn(self.norm1(x), rope_cos, rope_sin, return_attn=return_attn)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_weights


# ── Full EVA-02 Backbone ──────────────────────────────────────────────────────

class EVA02Backbone(BaseVisionBackbone):
    """EVA-02 backbone: ViT with 2D RoPE, SwiGLU, and Sub-LN.

    No CLS token — uses mean-pooling over patch tokens as the global descriptor.

    Output:
      - pooled_output:  mean(patch_tokens, dim=1) [B, D]
      - patch_tokens:   patch tokens [B, N, D]
      - feature_maps:   spatial reshape of patch tokens [B, D, H/p, W/p]
    """

    arch_name: str = "eva_02"

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
        self.feature_dims = [embed_dim]

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.grid_size = self.patch_embed.grid_size
        num_patches = self.patch_embed.num_patches

        # No CLS token in EVA-02 style — mean-pool patches at the end
        # No learned positional embedding — RoPE handles position encoding in Q/K

        self.blocks = nn.ModuleList([
            EVABlock(embed_dim, num_heads, mlp_ratio=mlp_ratio, attn_drop=attn_drop, mlp_drop=drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.drop = nn.Dropout(drop_rate)

        # Pre-compute RoPE frequencies for the fixed grid size
        # Registered as buffers so they move with .to(device) calls
        head_dim = embed_dim // num_heads
        cos_freq, sin_freq = _build_rope_freqs(self.grid_size, head_dim)
        self.register_buffer("rope_cos", cos_freq, persistent=False)  # [N, head_dim]
        self.register_buffer("rope_sin", sin_freq, persistent=False)

    def freeze_up_to_stage(self, stage: int) -> None:
        """Freeze patch embed and the first `stage` blocks."""
        self.patch_embed.requires_grad_(False)
        for i in range(min(stage + 1, self.depth)):
            self.blocks[i].requires_grad_(False)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        B = x.shape[0]

        # Patch embedding — no positional embedding added (RoPE handles positions)
        x = self.drop(self.patch_embed(x))              # [B, N, D]

        # Transformer blocks — pass RoPE frequencies to each block
        attn_maps: list[torch.Tensor] = []
        for blk in self.blocks:
            x, attn = blk(x, self.rope_cos, self.rope_sin, return_attn=self.return_all_attn)
            if self.return_all_attn and attn is not None:
                attn_maps.append(attn)

        x = self.norm(x)                                # [B, N, D]

        # Mean-pool over all patches as the global descriptor
        pooled = x.mean(dim=1)                          # [B, D]

        # Spatial reshape for dense tasks
        G = self.grid_size
        feat_map = x.transpose(1, 2).reshape(B, self.embed_dim, G, G)  # [B, D, G, G]

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=x,
            feature_maps=[feat_map],
            attentions=attn_maps if self.return_all_attn else None,
        )


# ── Variant factories ─────────────────────────────────────────────────────────

@register_vision_backbone("eva_02_small")
class EVA02Small(EVA02Backbone):
    """EVA-02-S/16 — 22M params."""
    arch_name = "eva_02_small"
    feature_dims = [384]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 384)
        kwargs.setdefault("depth", 12)
        kwargs.setdefault("num_heads", 6)
        kwargs.setdefault("patch_size", 16)
        super().__init__(**kwargs)


@register_vision_backbone("eva_02_base")
class EVA02Base(EVA02Backbone):
    """EVA-02-B/16 — 86M params."""
    arch_name = "eva_02_base"
    feature_dims = [768]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 768)
        kwargs.setdefault("depth", 12)
        kwargs.setdefault("num_heads", 12)
        kwargs.setdefault("patch_size", 16)
        super().__init__(**kwargs)


@register_vision_backbone("eva_02_large")
class EVA02Large(EVA02Backbone):
    """EVA-02-L/14 — 307M params.  The flagship EVA-02 model."""
    arch_name = "eva_02_large"
    feature_dims = [1024]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 1024)
        kwargs.setdefault("depth", 24)
        kwargs.setdefault("num_heads", 16)
        kwargs.setdefault("patch_size", 14)
        super().__init__(**kwargs)
