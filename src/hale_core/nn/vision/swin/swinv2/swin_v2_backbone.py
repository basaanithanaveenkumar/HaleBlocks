"""``SwinV2Backbone``: SwinV2 vision backbone."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.swin.swinv2._helpers import _trunc_normal_
from hale_core.nn.vision.swin.swinv2.patch_embed import PatchEmbed
from hale_core.nn.vision.swin.swinv2.swin_v2_stage import SwinV2Stage


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
        self.__class__.feature_dims = [embed_dim * (2**i) for i in range(len(depths))]
        self.__class__.embed_dim = embed_dim * (2 ** (len(depths) - 1))

        # Distribute stochastic depth linearly over all blocks
        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # Stem: 4×4 patch projection
        self.patch_embed = PatchEmbed(self.in_chans, embed_dim)

        # 4 stages
        block_idx = 0
        self.stages = nn.ModuleList()
        for i, (depth, heads) in enumerate(zip(depths, num_heads)):
            stage_rates = dp_rates[block_idx : block_idx + depth]
            block_idx += depth
            self.stages.append(
                SwinV2Stage(
                    dim=embed_dim * (2**i),  # Current stage's channel count
                    depth=depth,
                    num_heads=heads,
                    window_size=window_size,
                    downsample=(i > 0),  # No downsample at stage 0 (patch embed does it)
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
            patch_tokens=None,  # SwinV2 is spatial, not sequential
            feature_maps=feature_maps,
        )
