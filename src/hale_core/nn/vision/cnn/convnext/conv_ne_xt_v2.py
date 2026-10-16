"""``ConvNeXtV2``: Full ConvNeXt V2 feature extractor."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.cnn.convnext.conv_ne_xt_v2_block import ConvNeXtV2Block
from hale_core.nn.vision.cnn.convnext.layer_norm2d import LayerNorm2d


class ConvNeXtV2(BaseVisionBackbone):
    """Full ConvNeXt V2 feature extractor.

    Builds 4 stages of ConvNeXtV2Block with progressive downsampling,
    returning 4-scale feature maps for FPN/detection heads and a global
    pooled embedding for classification/VLM heads.

    Args:
        depths:      Number of blocks in each of the 4 stages.
        dims:        Channel dimensions at each of the 4 stages.
        drop_path_rate: Maximum stochastic depth rate (linearly distributed).
        img_size:    Expected input spatial resolution.
        in_chans:    Input image channels (default 3).
        drop_rate:   Dropout after global pool.
        out_indices: Which stage outputs to include in feature_maps.
    """

    def __init__(
        self,
        depths: list[int],
        dims: list[int],
        drop_path_rate: float = 0.0,
        img_size: int = 224,
        in_chans: int = 3,
        drop_rate: float = 0.0,
        out_indices: list[int] | None = None,
    ) -> None:
        super().__init__(
            img_size=img_size, in_chans=in_chans, drop_rate=drop_rate, out_indices=out_indices
        )

        self.embed_dim = dims[-1]
        self.feature_dims = list(dims)

        # Linearly distribute drop_path rates across all blocks
        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # ── Stem: patch-like tokenisation (4×4 stride-4) ─────────────────────
        # Unlike ViT's patch embedding, this is a simple large-stride conv + LN.
        # Reduces [B, 3, H, W] → [B, dims[0], H/4, W/4]
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6),
        )

        # ── 4 main stages ─────────────────────────────────────────────────────
        self.stages = nn.ModuleList()
        block_idx = 0
        for i, (depth, dim) in enumerate(zip(depths, dims)):
            blocks = [ConvNeXtV2Block(dim, drop_path=dp_rates[block_idx + j]) for j in range(depth)]
            self.stages.append(nn.Sequential(*blocks))
            block_idx += depth

        # ── Downsampling between stages (stages 0→1, 1→2, 2→3) ───────────────
        # LayerNorm + 2×2 stride-2 conv halves resolution and doubles channels
        self.downsamples = nn.ModuleList(
            [
                nn.Sequential(
                    LayerNorm2d(dims[i], eps=1e-6),
                    nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
                )
                for i in range(len(dims) - 1)
            ]
        )

        # ── Head ─────────────────────────────────────────────────────────────
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        """Truncated normal for conv/linear; const for norm layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """Forward pass returning multi-scale feature maps and pooled embedding.

        Args:
            x: [B, in_chans, H, W]

        Returns:
            BackboneOutput:
              pooled_output — [B, dims[-1]]
              patch_tokens  — None
              feature_maps  — 4 tensors at strides 4, 8, 16, 32
        """
        x = self.stem(x)  # [B, dims[0], H/4, W/4]

        feature_maps = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            feature_maps.append(x)  # collect output of each stage
            if i < len(self.downsamples):  # downsample between stages (not after last)
                x = self.downsamples[i](x)

        # Global pooled embedding
        pooled = self.pool(feature_maps[-1]).flatten(1)  # [B, dims[-1]]
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)

        # Select requested scales
        all_maps = feature_maps
        if self.out_indices is not None:
            all_maps = [feature_maps[i] for i in self.out_indices]

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=None,
            feature_maps=all_maps,
            attentions=None,
        )

    def freeze_up_to_stage(self, stage: int) -> None:
        """Freeze stem and stages 0 through `stage` inclusive.

        Args:
            stage: 0-based index in [0, 3].
        """
        if stage < 0 or stage > 3:
            raise ValueError(f"stage must be in [0, 3], got {stage}")
        for p in self.stem.parameters():
            p.requires_grad_(False)
        for i in range(stage + 1):
            for p in self.stages[i].parameters():
                p.requires_grad_(False)
            if i < len(self.downsamples):
                for p in self.downsamples[i].parameters():
                    p.requires_grad_(False)
