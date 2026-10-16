"""``SwinV2Block``: A single SwinV2 transformer block with post-norm."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.swin.swinv2._helpers import window_partition, window_reverse
from hale_core.nn.vision.swin.swinv2.drop_path import DropPath
from hale_core.nn.vision.swin.swinv2.swin_v2_window_attention import SwinV2WindowAttention


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
