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

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.registry import register_vision_backbone


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _trunc_normal_(tensor: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    """Truncated normal initialization (values clipped to ±2σ)."""
    with torch.no_grad():
        return tensor.normal_(0, std).clamp_(-2 * std, 2 * std)


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """Partition spatial feature map into non-overlapping windows.

    Args:
        x:           [B, H, W, C]
        window_size: Window height == width

    Returns:
        windows: [num_windows*B, window_size, window_size, C]
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # Reorder to [B, nH, nW, win, win, C] → [B*nH*nW, win, win, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, C)


def window_reverse(
    windows: torch.Tensor, window_size: int, H: int, W: int
) -> torch.Tensor:
    """Reconstruct feature map from windows.

    Args:
        windows:     [num_windows*B, window_size, window_size, C]
        window_size: Window height == width
        H, W:        Original spatial dimensions

    Returns:
        x: [B, H, W, C]
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


# ─── Log-CPB Relative Position Bias ──────────────────────────────────────────

class LogContinuousPositionBias(nn.Module):
    """Log-spaced Continuous Position Bias (Log-CPB) for SwinV2.

    Replaces the fixed learnable table of SwinV1 with a small MLP that inputs
    log-compressed coordinate differences. This allows inference at resolutions
    different from training (the MLP generalises; a table does not).

    The MLP maps (2,) → (num_heads,) via two linear layers.
    """

    def __init__(self, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        # Small MLP: 2 → 512 → num_heads (no bias on last layer for stability)
        self.cpb_mlp = nn.Sequential(
            nn.Linear(2, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_heads, bias=False),
        )
        # Cache the last computed bias grid for efficiency
        self._cached_grid_size: Optional[tuple] = None
        self._cached_bias: Optional[torch.Tensor] = None

    def _compute_bias(self, window_size: int) -> torch.Tensor:
        """Compute relative position bias for a given window size.

        Returns:
            bias: [num_heads, window_size*window_size, window_size*window_size]
        """
        coords_h = torch.arange(window_size, device=next(self.parameters()).device)
        coords_w = torch.arange(window_size, device=next(self.parameters()).device)
        # All pairwise (row, col) differences
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # [2, win, win]
        coords_flat = coords.flatten(1)  # [2, win*win]
        # Relative distances: [2, win*win, win*win]
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]
        rel = rel.permute(1, 2, 0).float()  # [win*win, win*win, 2]

        # Log-compress: sign(d) * log(1 + |d|)  — values in ≈ [-3.2, 3.2] for win=8
        rel_log = rel.sign() * torch.log1p(rel.abs())

        # Normalise to [-8, 8] so MLP sees consistent scale regardless of window_size
        norm_factor = math.log(window_size)
        rel_log = rel_log / norm_factor * 8.0

        # MLP → [win*win, win*win, num_heads]
        bias = self.cpb_mlp(rel_log)
        # Apply sigmoid to keep magnitudes bounded → scale to [-log(max_dist), log(max_dist)]
        bias = 16.0 * bias.sigmoid()
        # Rearrange to [num_heads, win*win, win*win]
        bias = bias.permute(2, 0, 1).contiguous()
        return bias

    def forward(self, window_size: int) -> torch.Tensor:
        """Return relative position bias for given window size (cached per size)."""
        if self._cached_grid_size != window_size:
            self._cached_bias = self._compute_bias(window_size)
            self._cached_grid_size = window_size
        return self._cached_bias  # type: ignore[return-value]


# ─── SwinV2 Attention ─────────────────────────────────────────────────────────

class SwinV2WindowAttention(nn.Module):
    """Scaled cosine self-attention within a single window.

    SwinV2 change: replaces dot-product Q·K^T with cosine similarity / tau
    to prevent logit explosion in large models.

    Args:
        dim:         Input channel count.
        window_size: Window height == width.
        num_heads:   Number of attention heads.
        attn_drop:   Dropout on attention weights.
        proj_drop:   Dropout on output projection.
    """

    def __init__(
        self,
        dim: int,
        window_size: int,
        num_heads: int,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        # Q, K, V projection (all-in-one for efficiency)
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # Log-CPB for resolution-agnostic relative position bias
        self.rel_pos_bias = LogContinuousPositionBias(num_heads)

        # Learnable per-head log-scale temperature (init=log(10) ≈ 2.30)
        # tau = exp(logit_scale.clip(max=log(100))) to prevent collapse
        self.logit_scale = nn.Parameter(torch.full((num_heads, 1, 1), math.log(10.0)))

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x:    [B*num_windows, win*win, C]
            mask: [num_windows, win*win, win*win] or None
        Returns:
            out: [B*num_windows, win*win, C]
        """
        Bw, N, C = x.shape
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(Bw, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, Bw, heads, N, head_dim]
        q, k, v = qkv.unbind(0)

        # Normalise Q and K to unit length for cosine similarity
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        # Temperature-scaled cosine attention
        tau = torch.clamp(self.logit_scale, max=math.log(100.0)).exp()
        attn = (q @ k.transpose(-2, -1)) * tau  # [Bw, heads, N, N]

        # Add Log-CPB relative position bias
        rpb = self.rel_pos_bias(self.window_size)  # [heads, N, N]
        attn = attn + rpb.unsqueeze(0)

        # Shifted-window attention mask: additive −∞ for cross-region pairs
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(Bw // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = self.attn_drop(F.softmax(attn, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(Bw, N, C)
        return self.proj_drop(self.proj(out))


# ─── SwinV2 Block ─────────────────────────────────────────────────────────────

class SwinV2Block(nn.Module):
    """A single SwinV2 transformer block with post-norm.

    Two variants controlled by `shift`:
      shift=False → standard window attention
      shift=True  → shifted window attention (cross-window context)

    SwinV2 change: LayerNorm is placed AFTER the residual connection
    (post-norm) instead of before (pre-norm). This improves gradient flow.

    Args:
        dim:         Channel dimension.
        num_heads:   Attention heads.
        window_size: Window spatial size.
        shift:       Whether to use shifted window attention.
        mlp_ratio:   FFN hidden dim ratio (default 4×).
        drop:        Dropout rate.
        attn_drop:   Attention dropout.
        drop_path:   Stochastic depth drop rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: int,
        shift: bool = False,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0

        self.attn = SwinV2WindowAttention(
            dim, window_size, num_heads, attn_drop=attn_drop, proj_drop=drop
        )
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )
        # Post-norm (SwinV2 change: norm goes AFTER residual, not before)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        # Stochastic depth: drop entire residual branch with probability drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        # Cached shifted-window attention mask (computed once per H,W)
        self._attn_mask: Optional[torch.Tensor] = None
        self._mask_hw: Optional[tuple] = None

    def _get_attn_mask(self, H: int, W: int, device: torch.device) -> Optional[torch.Tensor]:
        """Compute or retrieve cached attention mask for shifted windows.

        The mask ensures shifted-window tokens only attend within their
        original (pre-shift) window region — prevents cross-region mixing.

        Returns None for non-shifted blocks (no mask needed).
        """
        if self.shift_size == 0:
            return None
        if self._mask_hw == (H, W):
            return self._attn_mask

        # Label each window region with an integer ID
        img_mask = torch.zeros((1, H, W, 1), device=device)
        slices_h = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        slices_w = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for sh in slices_h:
            for sw in slices_w:
                img_mask[:, sh, sw, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, win, win, 1]
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        # Tokens with different region IDs should NOT attend to each other
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0)

        self._attn_mask = attn_mask
        self._mask_hw = (H, W)
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, H, W, C]
        Returns:
            x: [B, H, W, C]
        """
        B, H, W, C = x.shape

        # Pad to the nearest multiple of window_size so partition never sees a
        # remainder.  The padding is stripped before the residual connection so
        # the output always has the original H, W.
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x_in = F.pad(x, (0, 0, 0, pad_w, 0, pad_h)) if (pad_h or pad_w) else x
        Hp, Wp = H + pad_h, W + pad_w

        # Cyclic shift for shifted-window variant
        if self.shift_size > 0:
            shifted = torch.roll(x_in, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted = x_in

        # Partition into windows → [B*nW, win*win, C]
        win_x = window_partition(shifted, self.window_size)
        win_x = win_x.view(-1, self.window_size * self.window_size, C)

        # Window attention (mask is built for the padded Hp×Wp grid)
        mask = self._get_attn_mask(Hp, Wp, x.device)
        attn_out = self.attn(win_x, mask=mask)

        # Reconstruct spatial layout
        attn_out = attn_out.view(-1, self.window_size, self.window_size, C)
        shifted = window_reverse(attn_out, self.window_size, Hp, Wp)

        # Reverse cyclic shift
        if self.shift_size > 0:
            shifted = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

        # Strip padding before residual so shapes match the original x
        if pad_h or pad_w:
            shifted = shifted[:, :H, :W, :].contiguous()

        # Post-norm residual (SwinV2 change)
        x = self.norm1(x + self.drop_path(shifted))

        # FFN with post-norm residual
        x = self.norm2(x + self.drop_path(self.mlp(x)))
        return x


# ─── Stochastic Depth ────────────────────────────────────────────────────────

class DropPath(nn.Module):
    """Stochastic depth: drop entire residual branch per-sample during training.

    At inference (eval mode) acts as identity.  The remaining branches are
    scaled up by 1/(1-p) so expected value is preserved (like Dropout).

    Args:
        drop_prob: Probability of dropping a branch in a given step.
    """

    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        noise = torch.rand(shape, dtype=x.dtype, device=x.device).floor_() + keep_prob
        return x.div(keep_prob) * noise


# ─── Patch Embed & Merge ──────────────────────────────────────────────────────

class PatchEmbed(nn.Module):
    """4×4 non-overlapping patch projection (stem).

    Converts [B, 3, H, W] → [B, H/4, W/4, embed_dim].
    Uses a single Conv2d(kernel=4, stride=4) followed by LayerNorm.
    """

    def __init__(self, in_chans: int = 3, embed_dim: int = 96) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=4, stride=4)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] → [B, H/4, W/4, embed_dim]"""
        x = self.proj(x)           # [B, embed_dim, H/4, W/4]
        x = x.permute(0, 2, 3, 1) # [B, H/4, W/4, embed_dim] — channels-last
        return self.norm(x)


class PatchMerging(nn.Module):
    """Downsample by 2× and double channels: H×W×C → H/2×W/2×2C.

    Takes 2×2 spatial neighbours, concatenates channel-wise (→ 4C),
    applies LayerNorm, then projects 4C → 2C with a linear layer.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, H, W, C] → [B, H/2, W/2, 2C]"""
        B, H, W, C = x.shape
        # Sample every-other pixel in both spatial dimensions
        x0 = x[:, 0::2, 0::2, :]  # top-left
        x1 = x[:, 1::2, 0::2, :]  # bottom-left
        x2 = x[:, 0::2, 1::2, :]  # top-right
        x3 = x[:, 1::2, 1::2, :]  # bottom-right
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # [B, H/2, W/2, 4C]
        return self.reduction(self.norm(x))


# ─── SwinV2 Stage ─────────────────────────────────────────────────────────────

class SwinV2Stage(nn.Module):
    """One stage: optional PatchMerging + N SwinV2 blocks.

    Blocks alternate between regular (shift=False) and shifted (shift=True)
    window attention to capture both local and cross-window context.

    Args:
        dim:          Input channel count.
        depth:        Number of transformer blocks.
        num_heads:    Attention heads.
        window_size:  Window spatial size.
        downsample:   Whether to prepend a PatchMerging layer.
        drop_path_rates: List of stochastic-depth rates per block.
    """

    def __init__(
        self,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: int,
        downsample: bool = True,
        drop_path_rates: Optional[list] = None,
    ) -> None:
        super().__init__()
        # Optional spatial downsampling at the beginning of each stage
        self.downsample = PatchMerging(dim // 2) if downsample else nn.Identity()
        # Output dim after optional merge
        out_dim = dim if downsample else dim

        rates = drop_path_rates or [0.0] * depth
        self.blocks = nn.ModuleList([
            SwinV2Block(
                dim=out_dim,
                num_heads=num_heads,
                window_size=window_size,
                shift=(i % 2 == 1),  # Alternate: even=regular, odd=shifted
                drop_path=rates[i],
            )
            for i in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, H, W, C] → [B, H', W', C'] (C' doubled if downsample)"""
        x = self.downsample(x)
        for blk in self.blocks:
            x = blk(x)
        return x


# ─── SwinV2 Backbone ──────────────────────────────────────────────────────────

class SwinV2Backbone(BaseVisionBackbone):
    """SwinV2 vision backbone.

    Args:
        embed_dim:   Channel count after stem (stage 0 has this dim).
        depths:      Number of blocks per stage, e.g. [2, 2, 6, 2].
        num_heads:   Attention heads per stage, e.g. [3, 6, 12, 24].
        window_size: Local attention window size (default 8 for 224px, 12 for 384px).
        drop_path_rate: Total stochastic depth rate (linearly distributed over blocks).
        **base_kwargs: Passed to BaseVisionBackbone (img_size, in_chans, etc.)
    """

    def __init__(
        self,
        embed_dim: int = 96,
        depths: list = None,
        num_heads: list = None,
        window_size: int = 8,
        drop_path_rate: float = 0.2,
        **base_kwargs,
    ) -> None:
        super().__init__(**base_kwargs)
        depths = depths or [2, 2, 6, 2]
        num_heads = num_heads or [3, 6, 12, 24]

        # Set class-level attributes used by base class __repr__
        self.__class__.feature_dims = [embed_dim * (2 ** i) for i in range(len(depths))]
        self.__class__.embed_dim = embed_dim * (2 ** (len(depths) - 1))

        # Distribute stochastic depth linearly over all blocks
        total_blocks = sum(depths)
        dp_rates = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)
        ]

        # Stem: 4×4 patch projection
        self.patch_embed = PatchEmbed(self.in_chans, embed_dim)

        # 4 stages
        block_idx = 0
        self.stages = nn.ModuleList()
        for i, (depth, heads) in enumerate(zip(depths, num_heads)):
            stage_rates = dp_rates[block_idx: block_idx + depth]
            block_idx += depth
            self.stages.append(
                SwinV2Stage(
                    dim=embed_dim * (2 ** i),       # Current stage's channel count
                    depth=depth,
                    num_heads=heads,
                    window_size=window_size,
                    downsample=(i > 0),             # No downsample at stage 0 (patch embed does it)
                    drop_path_rates=stage_rates,
                )
            )

        self.norm = nn.LayerNorm(self.embed_dim)
        self.drop = nn.Dropout(self.drop_rate)

        self._init_weights()

    def _init_weights(self) -> None:
        """Truncated normal init for all linear/conv layers."""
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                _trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def freeze_up_to_stage(self, stage: int) -> None:
        """Freeze patch embed + stages 0..stage (inclusive)."""
        for p in self.patch_embed.parameters():
            p.requires_grad_(False)
        for i in range(min(stage + 1, len(self.stages))):
            for p in self.stages[i].parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            BackboneOutput with pooled_output and 4-scale feature_maps
        """
        # Stem: [B, 3, H, W] → [B, H/4, W/4, C]
        x = self.patch_embed(x)

        feature_maps = []
        for stage in self.stages:
            x = stage(x)
            # Convert channels-last → channels-first for feature_maps convention
            fm = x.permute(0, 3, 1, 2).contiguous()  # [B, C, H, W]
            feature_maps.append(fm)

        # Final norm on last stage features
        x = self.norm(x)

        # Global average pool over spatial dims for pooled embedding
        pooled = x.mean(dim=(1, 2))  # [B, C]
        pooled = self.drop(pooled)

        # Filter by out_indices if specified
        if self.out_indices is not None:
            feature_maps = [feature_maps[i] for i in self.out_indices]

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=None,          # SwinV2 is spatial, not sequential
            feature_maps=feature_maps,
        )


# ─── Registered Variants ─────────────────────────────────────────────────────

@register_vision_backbone("swinv2_tiny")
class SwinV2Tiny(SwinV2Backbone):
    """SwinV2-Tiny: 28M params. Fast, suitable for mobile / edge."""
    arch_name = "swinv2_tiny"
    embed_dim = 768   # 96 * 8

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 96)
        kwargs.setdefault("depths", [2, 2, 6, 2])
        kwargs.setdefault("num_heads", [3, 6, 12, 24])
        kwargs.setdefault("window_size", 8)
        super().__init__(**kwargs)


@register_vision_backbone("swinv2_small")
class SwinV2Small(SwinV2Backbone):
    """SwinV2-Small: 50M params. Good balance of accuracy/speed."""
    arch_name = "swinv2_small"
    embed_dim = 768

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 96)
        kwargs.setdefault("depths", [2, 2, 18, 2])
        kwargs.setdefault("num_heads", [3, 6, 12, 24])
        kwargs.setdefault("window_size", 8)
        super().__init__(**kwargs)


@register_vision_backbone("swinv2_base")
class SwinV2Base(SwinV2Backbone):
    """SwinV2-Base: 88M params. High accuracy, used in many benchmarks."""
    arch_name = "swinv2_base"
    embed_dim = 1024

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 128)
        kwargs.setdefault("depths", [2, 2, 18, 2])
        kwargs.setdefault("num_heads", [4, 8, 16, 32])
        kwargs.setdefault("window_size", 8)
        super().__init__(**kwargs)


@register_vision_backbone("swinv2_large")
class SwinV2Large(SwinV2Backbone):
    """SwinV2-Large: 197M params. Strong ImageNet-22K pretrain."""
    arch_name = "swinv2_large"
    embed_dim = 1536

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 192)
        kwargs.setdefault("depths", [2, 2, 18, 2])
        kwargs.setdefault("num_heads", [6, 12, 24, 48])
        kwargs.setdefault("window_size", 12)
        super().__init__(**kwargs)


@register_vision_backbone("swinv2_giant")
class SwinV2Giant(SwinV2Backbone):
    """SwinV2-Giant: 3B params. Highest accuracy in the SwinV2 family."""
    arch_name = "swinv2_giant"
    embed_dim = 1536

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 192)
        kwargs.setdefault("depths", [2, 2, 42, 4])
        kwargs.setdefault("num_heads", [6, 12, 24, 48])
        kwargs.setdefault("window_size", 12)
        super().__init__(**kwargs)
