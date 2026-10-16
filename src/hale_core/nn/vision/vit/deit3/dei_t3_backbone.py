"""``DeiT3Backbone``: DeiT III backbone: ViT with LayerScale and mean-pool (no CLS token)."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.vit.deit3.dei_t3_block import DeiT3Block
from hale_core.nn.vision.vit.vit import PatchEmbed, _trunc_normal_


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
        super().__init__(
            img_size=img_size, in_chans=in_chans, drop_rate=drop_rate, out_indices=out_indices
        )
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

        self.blocks = nn.ModuleList(
            [
                DeiT3Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    mlp_drop=drop_rate,
                    ls_init=ls_init,
                )
                for _ in range(depth)
            ]
        )

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
        x = self.patch_embed(x)  # [B, N, D]
        x = self.pos_drop(x + self.pos_embed)

        # 2. Transformer blocks
        attn_maps: list[torch.Tensor] = []
        for blk in self.blocks:
            x, attn = blk(x, return_attn=self.return_all_attn)
            if self.return_all_attn and attn is not None:
                attn_maps.append(attn)

        x = self.norm(x)  # [B, N, D]

        # 3. Mean pooling over all patches (no special CLS token)
        pooled = x.mean(dim=1)  # [B, D]

        # Spatial reshape for dense prediction heads
        G = self.patch_embed.grid_size
        feat_map = x.transpose(1, 2).reshape(B, self.embed_dim, G, G)  # [B, D, G, G]

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=x,
            feature_maps=[feat_map],
            attentions=attn_maps if self.return_all_attn else None,
        )
