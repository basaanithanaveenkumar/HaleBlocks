"""``DINOv2Backbone``: DINOv2 ViT backbone with register tokens and LayerScale."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.vit.dino.dino_block import DINOBlock

# Reuse PatchEmbed and MLP from sibling vit module (avoid duplication)
from hale_core.nn.vision.vit.vit import PatchEmbed, _trunc_normal_


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
        super().__init__(
            img_size=img_size, in_chans=in_chans, drop_rate=drop_rate, out_indices=out_indices
        )
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

        self.blocks = nn.ModuleList(
            [
                DINOBlock(
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
        patches = self.patch_embed(x)  # [B, N, D]

        # 2. Build token sequence: [CLS, patches] then add positional embedding
        if self.use_cls:
            cls = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
            tokens = torch.cat([cls, patches], dim=1)  # [B, 1+N, D]
        else:
            tokens = patches  # [B, N, D]

        tokens = self.pos_drop(tokens + self.pos_embed)

        # 3. Insert register tokens AFTER positional embedding (they don't have positions)
        regs = self.register_tokens.expand(B, -1, -1)  # [B, n_reg, D]
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
            n_prefix = 1 + self.n_register  # CLS + registers
            pooled = tokens[:, 0]  # CLS token [B, D]
            patches_out = tokens[:, n_prefix:]  # patch tokens [B, N, D]
        else:
            pooled = tokens[:, self.n_register :].mean(dim=1)  # mean pool over patches
            patches_out = tokens[:, self.n_register :]

        G = self.patch_embed.grid_size
        feat_map = patches_out.transpose(1, 2).reshape(B, self.embed_dim, G, G)

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=patches_out,
            feature_maps=[feat_map],
            attentions=attn_maps if self.return_all_attn else None,
        )
