"""``ConvNeXtV1``: Full ConvNeXt V1 feature extractor."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.cnn.convnext_v1.conv_ne_xt_v1_block import ConvNeXtV1Block
from hale_core.nn.vision.cnn.convnext_v1.layer_norm2d import LayerNorm2d


class ConvNeXtV1(BaseVisionBackbone):
    """Full ConvNeXt V1 feature extractor.

    Args:
        depths:                  Number of blocks in each of the 4 stages.
        dim:                     Base channel width (first stage).
        drop_path_rate:          Max stochastic depth rate (linearly distributed).
        layer_scale_init_value:  Initial layer scale γ (0 = disabled).
        img_size:                Expected input resolution.
        in_chans:                Input channels (default 3).
        drop_rate:               Dropout after global pool.
        out_indices:             Which stage feature maps to return.
    """

    def __init__(
        self,
        depths: list[int],
        dim: int,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: float = 0.0,
        img_size: int = 224,
        in_chans: int = 3,
        drop_rate: float = 0.0,
        out_indices: list[int] | None = None,
    ) -> None:
        super().__init__(
            img_size=img_size, in_chans=in_chans, drop_rate=drop_rate, out_indices=out_indices
        )

        dims = [dim, 2 * dim, 4 * dim, 8 * dim]
        self.embed_dim = dims[-1]
        self.feature_dims = dims

        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        # ── Stem: patchify (4×4, stride 4) ───────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm2d(dims[0], eps=1e-6),
        )

        # ── 4 main stages ─────────────────────────────────────────────────────
        self.stages = nn.ModuleList()
        block_idx = 0
        for depth, d in zip(depths, dims):
            blocks = [
                ConvNeXtV1Block(
                    dim=d,
                    drop_path=dp_rates[block_idx + j],
                    layer_scale_init_value=layer_scale_init_value,
                )
                for j in range(depth)
            ]
            self.stages.append(nn.Sequential(*blocks))
            block_idx += depth

        # ── Downsampling between stages ───────────────────────────────────────
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
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """Forward pass: stem → 4 stages with downsampling → pool.

        Returns:
            BackboneOutput with 4 feature maps at strides 4, 8, 16, 32.
        """
        x = self.stem(x)  # [B, dim, H/4, W/4]

        feature_maps = []
        for i, stage in enumerate(self.stages):
            x = stage(x)
            feature_maps.append(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        pooled = self.pool(feature_maps[-1]).flatten(1)  # [B, 8*dim]
        pooled = self.norm(pooled)
        pooled = self.dropout(pooled)

        if self.out_indices is not None:
            feature_maps = [feature_maps[i] for i in self.out_indices]

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=None,
            feature_maps=feature_maps,
            attentions=None,
        )

    def freeze_up_to_stage(self, stage: int) -> None:
        """Freeze stem and stages 0 through `stage` inclusive."""
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
