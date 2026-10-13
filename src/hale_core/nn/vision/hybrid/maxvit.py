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

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.registry import register_vision_backbone


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _trunc_normal_(t: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    with torch.no_grad():
        return t.normal_(0, std).clamp_(-2 * std, 2 * std)


class DropPath(nn.Module):
    """Stochastic depth — drop entire residual branch per sample."""

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.rand(shape, device=x.device, dtype=x.dtype).floor_() + keep
        return x.div(keep) * noise


# ─── Partition/Unpartition Helpers ───────────────────────────────────────────

def block_partition(x: torch.Tensor, block_size: int) -> torch.Tensor:
    """Partition spatial map into non-overlapping blocks.

    Each block of P×P pixels is one "window" for local attention.

    Args:
        x:          [B, H, W, C]  (channels-last)
        block_size: P — local window height == width

    Returns:
        blocks: [B * nH * nW, P, P, C]  where nH=H//P, nW=W//P
    """
    B, H, W, C = x.shape
    nH, nW = H // block_size, W // block_size
    x = x.view(B, nH, block_size, nW, block_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B * nH * nW, block_size, block_size, C)


def block_reverse(
    blocks: torch.Tensor, block_size: int, H: int, W: int
) -> torch.Tensor:
    """Reconstruct [B, H, W, C] from [B*nH*nW, P, P, C]."""
    nH, nW = H // block_size, W // block_size
    B = blocks.shape[0] // (nH * nW)
    x = blocks.view(B, nH, nW, block_size, block_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


def grid_partition(x: torch.Tensor, grid_size: int) -> torch.Tensor:
    """Partition into dilated grids for global attention.

    Instead of contiguous patches, takes every grid_size-th pixel.
    This gives each grid window a full-image receptive field at O(G²) cost.

    Example: 32×32 image, grid_size=8 → 8×8=64 windows, each 4×4 tokens.

    Args:
        x:         [B, H, W, C]
        grid_size: G — stride between picked pixels

    Returns:
        grids: [B * G * G, H//G, W//G, C]
    """
    B, H, W, C = x.shape
    gh, gw = H // grid_size, W // grid_size
    x = x.view(B, gh, grid_size, gw, grid_size, C)
    x = x.permute(0, 2, 4, 1, 3, 5).contiguous()
    return x.view(B * grid_size * grid_size, gh, gw, C)


def grid_reverse(
    grids: torch.Tensor, grid_size: int, H: int, W: int
) -> torch.Tensor:
    """Reconstruct [B, H, W, C] from [B*G*G, H//G, W//G, C]."""
    gh, gw = H // grid_size, W // grid_size
    B = grids.shape[0] // (grid_size * grid_size)
    x = grids.view(B, grid_size, grid_size, gh, gw, -1)
    return x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, H, W, -1)


# ─── Attention with Relative Position Bias ───────────────────────────────────

class WindowAttention(nn.Module):
    """Vanilla multi-head self-attention with 2D relative position bias.

    Used for both block and grid attention in MaxViT.
    The relative bias table is indexed by (row_diff, col_diff) pairs,
    similar to Swin but applied to a small fixed-size window.

    Args:
        dim:         Channel dimension.
        num_heads:   Attention heads.
        window_size: Spatial size of the attention window (H == W assumed).
        attn_drop:   Attention weight dropout.
        proj_drop:   Output projection dropout.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Relative position bias: table [2*win-1, 2*win-1, num_heads]
        table_size = (2 * window_size - 1) ** 2
        self.rel_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))
        _trunc_normal_(self.rel_bias_table, std=0.02)

        # Precompute relative position index for a window_size × window_size grid
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(window_size), torch.arange(window_size), indexing="ij"
            )
        ).flatten(1)  # [2, win*win]
        rel = coords[:, :, None] - coords[:, None, :]  # [2, N, N]
        rel = rel.permute(1, 2, 0)  # [N, N, 2]
        rel[:, :, 0] += window_size - 1
        rel[:, :, 1] += window_size - 1
        rel[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("rel_pos_index", rel.sum(-1))  # [N, N]

    def _build_rel_index(self, win: int, device: torch.device) -> torch.Tensor:
        """Compute relative position index for a win×win grid.

        Returns:
            index: [win*win, win*win] — each entry is a row-index into rel_bias_table
        """
        coords = torch.stack(
            torch.meshgrid(
                torch.arange(win, device=device),
                torch.arange(win, device=device),
                indexing="ij",
            )
        ).flatten(1)  # [2, win*win]
        rel = coords[:, :, None] - coords[:, None, :]  # [2, N, N]
        rel = rel.permute(1, 2, 0)                     # [N, N, 2]
        rel[:, :, 0] += win - 1
        rel[:, :, 1] += win - 1
        rel[:, :, 0] *= 2 * win - 1
        return rel.sum(-1)                              # [N, N]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B_win, N, C]  where N = window_size * window_size (may vary for grid attn)"""
        Bw, N, C = x.shape
        qkv = self.qkv(x).reshape(Bw, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # Relative position bias — recompute index dynamically when actual window
        # size differs from construction-time window_size (e.g. grid attention).
        win = int(N ** 0.5)
        if win * win == N and win == self.window_size:
            idx = self.rel_pos_index
        else:
            idx = self._build_rel_index(win, x.device)

        # Clamp indices so they stay in-bounds if the table was sized for a
        # larger window; out-of-range entries get the nearest valid bias.
        idx_clamped = idx.view(-1).clamp(0, self.rel_bias_table.shape[0] - 1)
        bias = self.rel_bias_table[idx_clamped].view(N, N, self.num_heads).permute(2, 0, 1)
        attn = attn + bias.unsqueeze(0)

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(Bw, N, C)
        return self.proj_drop(self.proj(out))


# ─── Squeeze-Excitation ───────────────────────────────────────────────────────

class SE(nn.Module):
    """Squeeze-Excite channel attention."""

    def __init__(self, ch: int, rd: int = None) -> None:
        super().__init__()
        rd = rd or max(1, ch // 4)
        self.fc1 = nn.Linear(ch, rd)
        self.fc2 = nn.Linear(rd, ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3))
        s = F.silu(self.fc1(s))
        s = self.fc2(s).sigmoid()
        return x * s[:, :, None, None]


# ─── MBConv Block ─────────────────────────────────────────────────────────────

class MBConv(nn.Module):
    """MBConv with BN pre-norm, depthwise conv and SE.

    Used in MaxViT as the first sub-layer of each MaxViT block for
    local feature extraction before the attention layers.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expand: int = 4,
        stride: int = 1,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        mid = in_ch * expand
        self.use_res = (in_ch == out_ch) and (stride == 1)
        self.pre_norm = nn.BatchNorm2d(in_ch)

        self.expand = nn.Conv2d(in_ch, mid, 1, bias=False)
        self.dw = nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False)
        self.dw_norm = nn.BatchNorm2d(mid)
        self.se = SE(mid)
        self.project = nn.Conv2d(mid, out_ch, 1, bias=False)
        self.proj_norm = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pre_norm(x)
        x = self.act(self.expand(x))
        x = self.act(self.dw_norm(self.dw(x)))
        x = self.se(x)
        x = self.proj_norm(self.project(x))
        if self.use_res:
            x = residual + self.drop_path(x)
        return x


# ─── Attention Sub-Layer (with FFN) ──────────────────────────────────────────

class AttentionLayer(nn.Module):
    """Pre-LN attention + FFN sub-layer used in MaxViT blocks.

    Layout:
      x → LN → attn(x) → + residual → LN → FFN → + residual

    Args:
        dim:         Channel dimension.
        num_heads:   Attention heads.
        window_size: Attention window size.
        mlp_ratio:   FFN expand ratio.
        drop_path:   Stochastic depth rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)

        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B_win, N, C]"""
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


# ─── MaxViT Block ─────────────────────────────────────────────────────────────

class MaxViTBlock(nn.Module):
    """Single MaxViT block: MBConv → Block Attention → Grid Attention.

    The three sub-layers give each block:
      - MBConv: strong local conv features (inductive bias)
      - Block Attention: fine-grained local self-attention
      - Grid Attention: global dilated self-attention

    Args:
        dim:          Channel dimension.
        num_heads:    Attention heads for both attention layers.
        block_size:   Local window size for block attention (default 8).
        mlp_ratio:    FFN expand ratio.
        drop_path:    Stochastic depth rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        block_size: int = 8,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.block_size = block_size

        # MBConv for local spatial features (operates in channels-first)
        self.mbconv = MBConv(dim, dim, drop_path=drop_path)

        # Block (local window) attention
        self.block_attn = AttentionLayer(
            dim, num_heads, block_size, mlp_ratio=mlp_ratio, drop_path=drop_path
        )

        # Grid (global dilated) attention  — same window_size=block_size
        # because grid_partition creates windows of size (H//block_size) × (W//block_size)
        # but we use block_size as the grid stride, producing block_size² windows
        self.grid_attn = AttentionLayer(
            dim, num_heads, block_size, mlp_ratio=mlp_ratio, drop_path=drop_path
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]  (channels-first)
        Returns:
            x: [B, C, H, W]
        """
        # ── MBConv (channels-first) ───────────────────────────────────────────
        x = self.mbconv(x)

        B, C, H, W = x.shape
        P = self.block_size

        # Convert to channels-last for attention operations
        x = x.permute(0, 2, 3, 1)  # [B, H, W, C]

        # Pad to a multiple of block_size so partition math always divides evenly.
        # Both block attention (P×P windows) and grid attention (stride-P sampling)
        # require H and W to be multiples of P.  The pad is removed at the end.
        pad_h = (P - H % P) % P
        pad_w = (P - W % P) % P
        if pad_h or pad_w:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        Hp, Wp = H + pad_h, W + pad_w

        # ── Block (local) Attention ───────────────────────────────────────────
        blocks = block_partition(x, P)              # [B*nH*nW, P, P, C]
        N = P * P
        blocks = blocks.view(-1, N, C)
        blocks = self.block_attn(blocks)
        blocks = blocks.view(-1, P, P, C)
        x = block_reverse(blocks, P, Hp, Wp)        # [B, Hp, Wp, C]

        # ── Grid (global) Attention ───────────────────────────────────────────
        # grid_partition samples every P-th location; each resulting window has
        # (Hp//P) × (Wp//P) tokens — size differs from P×P for block attention,
        # so WindowAttention uses a dynamic relative-bias index (see above).
        grids = grid_partition(x, P)                # [B*P*P, Hp//P, Wp//P, C]
        gh, gw = Hp // P, Wp // P
        grids = grids.view(-1, gh * gw, C)
        grids = self.grid_attn(grids)
        grids = grids.view(-1, gh, gw, C)
        x = grid_reverse(grids, P, Hp, Wp)          # [B, Hp, Wp, C]

        # Strip padding before returning
        if pad_h or pad_w:
            x = x[:, :H, :W, :].contiguous()

        # Back to channels-first
        return x.permute(0, 3, 1, 2).contiguous()


# ─── MaxViT Stage ─────────────────────────────────────────────────────────────

class MaxViTStage(nn.Module):
    """One MaxViT stage: optional strided MBConv downsample + N MaxViT blocks.

    Args:
        in_ch:     Input channels.
        out_ch:    Output channels.
        depth:     Number of MaxViT blocks.
        num_heads: Attention heads.
        block_size: Local window size.
        drop_path_rates: Per-block stochastic depth rates.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        depth: int,
        num_heads: int,
        block_size: int = 8,
        drop_path_rates: Optional[list] = None,
    ) -> None:
        super().__init__()
        rates = drop_path_rates or [0.0] * depth

        # Strided MBConv to halve resolution and adjust channels at stage entry
        self.downsample = MBConv(in_ch, out_ch, stride=2, drop_path=rates[0])

        # Remaining MaxViT blocks (all same channel count, stride=1)
        self.blocks = nn.ModuleList([
            MaxViTBlock(out_ch, num_heads, block_size=block_size, drop_path=rates[i + 1])
            for i in range(depth - 1)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x


# ─── MaxViT Backbone ──────────────────────────────────────────────────────────

class MaxViTBackbone(BaseVisionBackbone):
    """MaxViT vision backbone.

    4-stage hierarchy. Each stage: strided MBConv + N MaxViT blocks.
    Block attention window is block_size×block_size (default 8).
    Grid attention uses the same window size as dilated stride.

    Args:
        dim:         Base channel count (stage 0 has this many channels).
        depths:      Number of blocks per stage [d0, d1, d2, d3].
        block_size:  Local attention window size (must divide feature map size).
        drop_path_rate: Total stochastic depth drop rate.
        **base_kwargs: Passed to BaseVisionBackbone.
    """

    def __init__(
        self,
        dim: int = 64,
        depths: list = None,
        block_size: int = 8,
        drop_path_rate: float = 0.2,
        **base_kwargs,
    ) -> None:
        super().__init__(**base_kwargs)
        depths = depths or [2, 2, 5, 2]
        stage_dims = [dim * (2 ** i) for i in range(len(depths))]

        # Class-level attributes
        self.__class__.feature_dims = stage_dims
        self.__class__.embed_dim = stage_dims[-1]

        # Distribute stochastic depth rates
        total = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total)]
        idx = 0

        def _take(n: int) -> list:
            nonlocal idx
            r = dp_rates[idx: idx + n]
            idx += n
            return r

        # Stem: two 3×3 convs (stride-2 + stride-1)  input H → H/2
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_chans, dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        # 4 stages: each halves H and doubles C
        # Stage input channels: [dim, dim, dim*2, dim*4]  (stem produces dim, stage 0 doubles)
        stem_ch = dim
        self.stages = nn.ModuleList()
        in_ch = stem_ch
        for i, (d, out_ch) in enumerate(zip(depths, stage_dims)):
            heads = max(1, out_ch // 32)  # ~1 head per 32 channels
            self.stages.append(
                MaxViTStage(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    depth=d,
                    num_heads=heads,
                    block_size=block_size,
                    drop_path_rates=_take(d),
                )
            )
            in_ch = out_ch

        self.norm = nn.BatchNorm2d(stage_dims[-1])
        self.drop = nn.Dropout(self.drop_rate)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                _trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def freeze_up_to_stage(self, stage: int) -> None:
        """Freeze stem + stages 0..stage (inclusive)."""
        for p in self.stem.parameters():
            p.requires_grad_(False)
        for i in range(min(stage + 1, len(self.stages))):
            for p in self.stages[i].parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            BackboneOutput with 4 feature maps and pooled_output
        """
        x = self.stem(x)            # [B, dim, H/2, W/2]

        feature_maps = []
        for stage in self.stages:
            x = stage(x)            # Each stage halves H,W and doubles C
            feature_maps.append(x)

        x = self.norm(x)
        pooled = x.mean(dim=(2, 3))  # [B, stage_dims[-1]]
        pooled = self.drop(pooled)

        if self.out_indices is not None:
            feature_maps = [feature_maps[i] for i in self.out_indices]

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=None,
            feature_maps=feature_maps,
        )


# ─── Registered Variants ─────────────────────────────────────────────────────

@register_vision_backbone("maxvit_tiny")
class MaxViTTiny(MaxViTBackbone):
    """MaxViT-Tiny: ~31M params. Fastest MaxViT variant."""
    arch_name = "maxvit_tiny"
    embed_dim = 512   # 64 * 8

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("dim", 64)
        kwargs.setdefault("depths", [2, 2, 5, 2])
        super().__init__(**kwargs)


@register_vision_backbone("maxvit_small")
class MaxViTSmall(MaxViTBackbone):
    """MaxViT-Small: ~69M params. Good accuracy/speed trade-off."""
    arch_name = "maxvit_small"
    embed_dim = 768   # 96 * 8

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("dim", 96)
        kwargs.setdefault("depths", [2, 2, 5, 2])
        super().__init__(**kwargs)


@register_vision_backbone("maxvit_base")
class MaxViTBase(MaxViTBackbone):
    """MaxViT-Base: ~120M params. Higher accuracy with deeper stages."""
    arch_name = "maxvit_base"
    embed_dim = 768

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("dim", 96)
        kwargs.setdefault("depths", [2, 6, 14, 2])
        super().__init__(**kwargs)


@register_vision_backbone("maxvit_large")
class MaxViTLarge(MaxViTBackbone):
    """MaxViT-Large: ~212M params. Strong ImageNet accuracy."""
    arch_name = "maxvit_large"
    embed_dim = 1024   # 128 * 8

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("dim", 128)
        kwargs.setdefault("depths", [2, 6, 14, 2])
        super().__init__(**kwargs)


@register_vision_backbone("maxvit_xlarge")
class MaxViTXLarge(MaxViTBackbone):
    """MaxViT-XLarge: ~475M params. Largest MaxViT for high-res tasks."""
    arch_name = "maxvit_xlarge"
    embed_dim = 1536   # 192 * 8

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("dim", 192)
        kwargs.setdefault("depths", [2, 6, 14, 2])
        super().__init__(**kwargs)
