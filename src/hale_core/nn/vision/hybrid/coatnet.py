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


# ─── Squeeze-Excitation ───────────────────────────────────────────────────────

class SqueezeExcitation(nn.Module):
    """Channel-wise gating: global avg pool → 2-FC → sigmoid scale.

    Args:
        channels:    Input/output channel count.
        rd_ratio:    Reduction ratio for bottleneck (default 0.25 = 4×).
        activation:  Activation between FC layers.
    """

    def __init__(
        self,
        channels: int,
        rd_ratio: float = 0.25,
        activation: nn.Module = None,
    ) -> None:
        super().__init__()
        rd = max(1, int(channels * rd_ratio))
        self.fc1 = nn.Linear(channels, rd, bias=True)
        self.fc2 = nn.Linear(rd, channels, bias=True)
        self.act = activation or nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] → scale [B, C, 1, 1] applied to x."""
        s = x.mean(dim=(2, 3))          # Global avg pool: [B, C]
        s = self.act(self.fc1(s))       # Bottleneck
        s = self.fc2(s).sigmoid()       # Gate in [0, 1]
        return x * s[:, :, None, None]  # Scale channels


# ─── MBConv Block ─────────────────────────────────────────────────────────────

class MBConvBlock(nn.Module):
    """Mobile Inverted Bottleneck Convolution block.

    Used in S1 and S2 of CoAtNet. Pattern:
      [input]
        ↓ (BatchNorm — pre-norm applied to input)
        ↓ expand 1×1 conv (expand_ratio × C channels)
        ↓ depthwise 3×3 conv (groups=out_channels for true depthwise)
        ↓ Squeeze-Excitation
        ↓ project 1×1 conv (back to out_channels)
        ↓ (+ residual if in_channels == out_channels and stride==1)

    Args:
        in_ch:        Input channels.
        out_ch:       Output channels.
        stride:       Spatial stride (1 or 2).
        expand_ratio: Expansion multiplier for intermediate channels.
        se_ratio:     SE reduction ratio.
        drop_path:    Stochastic depth rate.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        expand_ratio: int = 4,
        se_ratio: float = 0.25,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.use_residual = (in_ch == out_ch and stride == 1)

        # Pre-BatchNorm normalises the input before the block
        self.pre_norm = nn.BatchNorm2d(in_ch)

        self.expand = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.expand_act = nn.SiLU()

        # Depthwise 3×3 — groups=mid_ch means each channel has its own filter
        self.dw_conv = nn.Conv2d(
            mid_ch, mid_ch, 3, stride=stride, padding=1, groups=mid_ch, bias=False
        )
        self.dw_bn = nn.BatchNorm2d(mid_ch)
        self.dw_act = nn.SiLU()

        self.se = SqueezeExcitation(mid_ch, rd_ratio=se_ratio / expand_ratio)

        self.project = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.project_bn = nn.BatchNorm2d(out_ch)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pre_norm(x)
        x = self.expand_act(self.expand(x))
        x = self.dw_act(self.dw_bn(self.dw_conv(x)))
        x = self.se(x)
        x = self.project_bn(self.project(x))
        if self.use_residual:
            x = residual + self.drop_path(x)
        return x


# ─── Relative Position Bias Table ────────────────────────────────────────────

class RelativePositionBias(nn.Module):
    """Learnable relative position bias indexed by (row_diff, col_diff).

    Table size: (2*max_dist-1) × (2*max_dist-1) per head.
    At each attention step, bias is indexed by the pair-wise position
    differences (clipped to [-max_dist+1, max_dist-1]).

    Args:
        num_heads: Number of attention heads.
        max_dist:  Maximum position difference to represent (default 64).
    """

    def __init__(self, num_heads: int, max_dist: int = 64) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.max_dist = max_dist
        table_size = (2 * max_dist - 1) ** 2
        self.table = nn.Parameter(torch.zeros(table_size, num_heads))
        _trunc_normal_(self.table, std=0.02)

        # Cache the index lookup for a given H×W grid
        self._cached_hw: Optional[tuple] = None
        self._cached_idx: Optional[torch.Tensor] = None

    def _build_index(self, H: int, W: int) -> torch.Tensor:
        """Precompute flat indices into the bias table for an H×W grid."""
        coords = torch.stack(
            torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        ).flatten(1)  # [2, H*W]
        # Pairwise differences
        rel_h = coords[0][:, None] - coords[0][None, :]  # [H*W, H*W]
        rel_w = coords[1][:, None] - coords[1][None, :]
        # Clip to [-max_dist+1, max_dist-1]
        md = self.max_dist - 1
        rel_h = rel_h.clamp(-md, md) + md
        rel_w = rel_w.clamp(-md, md) + md
        # Map to flat index into table of shape [2*max_dist-1, 2*max_dist-1]
        idx = rel_h * (2 * self.max_dist - 1) + rel_w  # [H*W, H*W]
        return idx

    def forward(self, H: int, W: int) -> torch.Tensor:
        """Return [num_heads, H*W, H*W] bias to add to attention logits."""
        if self._cached_hw != (H, W):
            self._cached_idx = self._build_index(H, W).to(self.table.device)
            self._cached_hw = (H, W)
        idx = self._cached_idx  # [H*W, H*W]
        N = H * W
        # Index table [table_size, num_heads] → [H*W, H*W, num_heads]
        bias = self.table[idx.view(-1)].view(N, N, self.num_heads)
        return bias.permute(2, 0, 1)  # [num_heads, H*W, H*W]


# ─── Relative Attention Block ─────────────────────────────────────────────────

class RelativeAttentionBlock(nn.Module):
    """Transformer block with relative position bias (used in S3, S4).

    Layout (pre-norm):
      x → LN → MHSA → + residual → LN → FFN → + residual

    Relative position bias is added to QK^T before softmax.

    Args:
        dim:       Channel dimension.
        num_heads: Attention heads.
        mlp_ratio: FFN hidden expand ratio.
        drop_path: Stochastic depth rate.
        max_dist:  Maximum relative position distance for bias table.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        max_dist: int = 64,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, dim),
        )

        self.rel_bias = RelativePositionBias(num_heads, max_dist=max_dist)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def _attn(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """Full self-attention with relative position bias over H×W spatial grid."""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # [B, heads, N, d]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # Add relative position bias [heads, N, N]
        attn = attn + self.rel_bias(H, W).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(out)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            x: [B, H*W, C]  — flattened spatial tokens
        Returns:
            x: [B, H*W, C]
        """
        x = x + self.drop_path(self._attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


# ─── CoAtNet Stage ────────────────────────────────────────────────────────────

class CoAtNetConvStage(nn.Module):
    """Sequence of MBConv blocks with optional downsampling at the start.

    Args:
        in_ch:    Input channels.
        out_ch:   Output channels.
        depth:    Number of MBConv blocks.
        stride:   Spatial stride for the first block (1 = no downscale).
        drop_path_rates: Per-block stochastic depth rates.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        depth: int,
        stride: int = 2,
        drop_path_rates: Optional[list] = None,
    ) -> None:
        super().__init__()
        rates = drop_path_rates or [0.0] * depth
        # First block handles channel change + optional stride
        blocks = [MBConvBlock(in_ch, out_ch, stride=stride, drop_path=rates[0])]
        for i in range(1, depth):
            blocks.append(MBConvBlock(out_ch, out_ch, stride=1, drop_path=rates[i]))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class CoAtNetAttnStage(nn.Module):
    """Sequence of Transformer blocks with optional spatial downsampling.

    Downsampling: MaxPool2d(2) + 1×1 Conv before the transformer blocks.

    Args:
        in_ch:    Input channels.
        out_ch:   Output channels.
        depth:    Number of transformer blocks.
        num_heads: Attention heads.
        stride:   Spatial stride (2 = halve resolution with pool+conv).
        drop_path_rates: Per-block stochastic depth rates.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        depth: int,
        num_heads: int,
        stride: int = 2,
        drop_path_rates: Optional[list] = None,
    ) -> None:
        super().__init__()
        rates = drop_path_rates or [0.0] * depth

        # Downsampling: pool for spatial halving, 1×1 conv for channel change
        if stride == 2:
            self.downsample = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.downsample = nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()

        self.blocks = nn.ModuleList([
            RelativeAttentionBlock(out_ch, num_heads, drop_path=rates[i])
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] → [B, C', H', W']"""
        x = self.downsample(x)
        B, C, H, W = x.shape
        # Flatten spatial dims for transformer: [B, H*W, C]
        tokens = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        for blk in self.blocks:
            tokens = blk(tokens, H, W)
        tokens = self.norm(tokens)
        # Restore spatial layout: [B, C, H, W]
        return tokens.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()


# ─── CoAtNet Backbone ─────────────────────────────────────────────────────────

class CoAtNetBackbone(BaseVisionBackbone):
    """CoAtNet vision backbone.

    5-stage architecture: Conv Stem + 2 MBConv stages + 2 Attention stages.

    Args:
        dims:    Channel counts for [stem, S1, S2, S3, S4].
        depths:  Block counts for [stem, S1, S2, S3, S4].
                 Stem depth is number of conv layers (typically 2).
        num_heads_attn: Attention heads for S3 and S4.
        drop_path_rate: Total stochastic depth rate.
        **base_kwargs: Passed to BaseVisionBackbone.
    """

    def __init__(
        self,
        dims: list = None,
        depths: list = None,
        num_heads_attn: int = 8,
        drop_path_rate: float = 0.2,
        **base_kwargs,
    ) -> None:
        super().__init__(**base_kwargs)
        dims = dims or [64, 96, 192, 384, 768]
        depths = depths or [2, 3, 5, 2, 2]

        # Class-level attributes for base __repr__
        self.__class__.feature_dims = dims[1:]       # S1..S4 channel counts
        self.__class__.embed_dim = dims[-1]

        # Distribute drop_path evenly over all non-stem blocks
        total = sum(depths[1:])
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total)]
        idx = 0

        # S0: Conv stem — two stride-2 convolutions  (input H → H/4)
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_chans, dims[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
            nn.Conv2d(dims[0], dims[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        )

        def _take(n: int) -> list:
            nonlocal idx
            r = dp_rates[idx: idx + n]
            idx += n
            return r

        # S1: MBConv blocks, stride-2 → H/8
        self.s1 = CoAtNetConvStage(
            dims[0], dims[1], depths[1], stride=2, drop_path_rates=_take(depths[1])
        )
        # S2: MBConv blocks, stride-2 → H/16
        self.s2 = CoAtNetConvStage(
            dims[1], dims[2], depths[2], stride=2, drop_path_rates=_take(depths[2])
        )
        # S3: Attention blocks, stride-2 → H/32
        heads3 = max(1, dims[3] // 64)  # ~1 head per 64 dim
        self.s3 = CoAtNetAttnStage(
            dims[2], dims[3], depths[3], num_heads=heads3, stride=2,
            drop_path_rates=_take(depths[3])
        )
        # S4: Attention blocks, stride-1 (maintain resolution)
        heads4 = max(1, dims[4] // 64)
        self.s4 = CoAtNetAttnStage(
            dims[3], dims[4], depths[4], num_heads=heads4, stride=1,
            drop_path_rates=_take(depths[4])
        )

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
        """Freeze stem (stage -1) + conv/attn stages 0..stage."""
        all_stages = [self.stem, self.s1, self.s2, self.s3, self.s4]
        for i in range(min(stage + 2, len(all_stages))):  # +2: stem counts as stage -1
            for p in all_stages[i].parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            BackboneOutput with 4 feature maps (S1..S4) and pooled_output
        """
        x = self.stem(x)          # [B, dims[0], H/4, W/4]

        s1 = self.s1(x)           # [B, dims[1], H/8,  W/8]
        s2 = self.s2(s1)          # [B, dims[2], H/16, W/16]
        s3 = self.s3(s2)          # [B, dims[3], H/32, W/32]
        s4 = self.s4(s3)          # [B, dims[4], H/32, W/32] (stride-1)

        # Global average pool over final spatial map for pooled embedding
        pooled = s4.mean(dim=(2, 3))  # [B, dims[4]]
        pooled = self.drop(pooled)

        feature_maps = [s1, s2, s3, s4]
        if self.out_indices is not None:
            feature_maps = [feature_maps[i] for i in self.out_indices]

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=None,
            feature_maps=feature_maps,
        )


# ─── Registered Variants ─────────────────────────────────────────────────────

@register_vision_backbone("coatnet_0")
class CoAtNet0(CoAtNetBackbone):
    """CoAtNet-0: ~25M params. Fastest in the CoAtNet family."""
    arch_name = "coatnet_0"
    embed_dim = 768

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("dims", [64, 96, 192, 384, 768])
        kwargs.setdefault("depths", [2, 3, 5, 2, 2])
        super().__init__(**kwargs)


@register_vision_backbone("coatnet_1")
class CoAtNet1(CoAtNetBackbone):
    """CoAtNet-1: ~42M params. More blocks for better feature quality."""
    arch_name = "coatnet_1"
    embed_dim = 768

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("dims", [64, 96, 192, 384, 768])
        kwargs.setdefault("depths", [2, 6, 14, 2, 2])
        super().__init__(**kwargs)


@register_vision_backbone("coatnet_2")
class CoAtNet2(CoAtNetBackbone):
    """CoAtNet-2: ~75M params. Wider with 128-dim stem."""
    arch_name = "coatnet_2"
    embed_dim = 1024

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("dims", [128, 128, 256, 512, 1024])
        kwargs.setdefault("depths", [2, 6, 14, 2, 2])
        super().__init__(**kwargs)


@register_vision_backbone("coatnet_3")
class CoAtNet3(CoAtNetBackbone):
    """CoAtNet-3: ~168M params. Large-scale model for high accuracy."""
    arch_name = "coatnet_3"
    embed_dim = 1536

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("dims", [192, 192, 384, 768, 1536])
        kwargs.setdefault("depths", [2, 6, 14, 2, 2])
        super().__init__(**kwargs)


@register_vision_backbone("coatnet_7")
class CoAtNet7(CoAtNetBackbone):
    """CoAtNet-7: ~2.4B params. Giant model matching ViT-22B scale."""
    arch_name = "coatnet_7"
    embed_dim = 1536

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("dims", [192, 192, 384, 768, 1536])
        kwargs.setdefault("depths", [2, 12, 28, 2, 2])
        super().__init__(**kwargs)
