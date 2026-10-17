"""``ViTBackbone``: Standard Vision Transformer backbone."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.vit.vit._helpers import _trunc_normal_
from hale_core.nn.vision.vit.vit.patch_embed import PatchEmbed
from hale_core.nn.vision.vit.vit.vi_t_block import ViTBlock


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
        super().__init__(
            img_size=img_size, in_chans=in_chans, drop_rate=drop_rate, out_indices=out_indices
        )
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
        self.blocks = nn.ModuleList(
            [
                ViTBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    mlp_drop=drop_rate,
                )
                for _ in range(depth)
            ]
        )

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
        """Forward pass of :class:`ViTBackbone`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``BackboneOutput``.
        """
        B = x.shape[0]

        # 1. Patch embedding
        x = self.patch_embed(x)  # [B, N, D]

        # 2. Prepend CLS token and add positional embedding
        cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat([cls, x], dim=1)  # [B, N+1, D]
        x = self.pos_drop(x + self.pos_embed)

        # 3. Transformer blocks
        attn_maps: list[torch.Tensor] = []
        for blk in self.blocks:
            x, attn = blk(x, return_attn=self.return_all_attn)
            if self.return_all_attn and attn is not None:
                attn_maps.append(attn)

        x = self.norm(x)  # [B, N+1, D]

        # 4. Unpack outputs
        pooled = x[:, 0]  # [B, D]  — CLS token
        patches = x[:, 1:]  # [B, N, D]

        # Reshape to spatial grid for dense tasks
        G = self.patch_embed.grid_size
        feat_map = patches.transpose(1, 2).reshape(B, self.embed_dim, G, G)  # [B, D, G, G]

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=patches,
            feature_maps=[feat_map],
            attentions=attn_maps if self.return_all_attn else None,
        )
