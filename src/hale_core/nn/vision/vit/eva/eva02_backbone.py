"""``EVA02Backbone``: EVA-02 backbone: ViT with 2D RoPE, SwiGLU, and Sub-LN."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.vit.eva._helpers import _build_rope_freqs
from hale_core.nn.vision.vit.eva.eva_block import EVABlock
from hale_core.nn.vision.vit.vit import PatchEmbed


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
        super().__init__(
            img_size=img_size, in_chans=in_chans, drop_rate=drop_rate, out_indices=out_indices
        )
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.depth = depth
        self.return_all_attn = return_all_attn
        self.feature_dims = [embed_dim]

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.grid_size = self.patch_embed.grid_size

        # No CLS token in EVA-02 style — mean-pool patches at the end
        # No learned positional embedding — RoPE handles position encoding in Q/K

        self.blocks = nn.ModuleList(
            [
                EVABlock(
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
        """Forward pass of :class:`EVA02Backbone`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``BackboneOutput``.
        """
        B = x.shape[0]

        # Patch embedding — no positional embedding added (RoPE handles positions)
        x = self.drop(self.patch_embed(x))  # [B, N, D]

        # Transformer blocks — pass RoPE frequencies to each block
        attn_maps: list[torch.Tensor] = []
        for blk in self.blocks:
            x, attn = blk(x, self.rope_cos, self.rope_sin, return_attn=self.return_all_attn)
            if self.return_all_attn and attn is not None:
                attn_maps.append(attn)

        x = self.norm(x)  # [B, N, D]

        # Mean-pool over all patches as the global descriptor
        pooled = x.mean(dim=1)  # [B, D]

        # Spatial reshape for dense tasks
        G = self.grid_size
        feat_map = x.transpose(1, 2).reshape(B, self.embed_dim, G, G)  # [B, D, G, G]

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=x,
            feature_maps=[feat_map],
            attentions=attn_maps if self.return_all_attn else None,
        )
