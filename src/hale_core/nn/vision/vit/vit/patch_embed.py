"""``PatchEmbed``: Convert an image into non-overlapping patch embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.vit.vit._helpers import _trunc_normal_


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
        self.grid_size = img_size // patch_size  # e.g. 14 for 224/16
        self.num_patches = self.grid_size**2

        # A single strided convolution performs the patch extraction + linear projection
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        _trunc_normal_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`PatchEmbed`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        # x: [B, C, H, W] → [B, D, H/p, W/p] → [B, N, D]
        x = self.proj(x)  # [B, D, G, G]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x
