"""``NFNet``: NFNet: Normalizer-Free high-performance backbone."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.cnn.nfnet._helpers import _NFNET_DEPTHS, _NFNET_WIDTHS
from hale_core.nn.vision.cnn.nfnet.nf_block import NFBlock
from hale_core.nn.vision.cnn.nfnet.ws_conv2d import WSConv2d


class NFNet(BaseVisionBackbone):
    """NFNet: Normalizer-Free high-performance backbone.

    No BatchNorm anywhere.  Weight standardisation + variance preservation
    residuals + adaptive gradient clipping at training time.

    Args:
        variant:        One of 'f0'..'f5' controlling depth multipliers.
        drop_path_rate: Max stochastic depth rate (linearly distributed).
        img_size:       Input spatial resolution.
        in_chans:       Input channels (default 3).
        drop_rate:      Dropout after global pool.
        out_indices:    Which stage feature maps to return.
    """

    embed_dim = 1536
    feature_dims = [256, 512, 1536, 1536]

    def __init__(
        self,
        variant: str = "f0",
        drop_path_rate: float = 0.0,
        img_size: int = 224,
        in_chans: int = 3,
        drop_rate: float = 0.0,
        out_indices: list[int] | None = None,
    ) -> None:
        super().__init__(
            img_size=img_size, in_chans=in_chans, drop_rate=drop_rate, out_indices=out_indices
        )

        depths = _NFNET_DEPTHS[variant]
        widths = _NFNET_WIDTHS

        # ── Stem: 4 WSConv layers → 4× spatial reduction ─────────────────────
        # Each stride-2 conv halves spatial dims; two stride-2 = 4× total.
        self.stem = nn.Sequential(
            WSConv2d(in_chans, 16, 3, stride=2, padding=1, bias=False),
            nn.GELU(),
            WSConv2d(16, 32, 3, stride=1, padding=1, bias=False),
            nn.GELU(),
            WSConv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.GELU(),
            WSConv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.GELU(),
        )

        # ── Build 4 stages ────────────────────────────────────────────────────
        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]

        self.stages = nn.ModuleList()
        block_idx = 0
        # Tracks expected std for variance-preservation beta calculation
        expected_std = 1.0
        in_ch = 128

        for stage_idx, (depth, width) in enumerate(zip(depths, widths)):
            blocks = []
            for i in range(depth):
                stride = 2 if (i == 0 and stage_idx > 0) else 1
                # beta = current expected std; resets to 1 if this block downsamples
                beta = expected_std
                blocks.append(
                    NFBlock(
                        in_channels=in_ch,
                        out_channels=width,
                        stride=stride,
                        group_size=128,
                        beta=beta,
                        alpha=0.2,
                        squeeze_ratio=0.5,
                        drop_path=dp_rates[block_idx],
                    )
                )
                in_ch = width
                block_idx += 1
                # Variance accumulates across blocks: std grows by sqrt(1 + alpha^2)
                expected_std = math.sqrt(expected_std**2 + 0.2**2)
                if stride > 1:
                    expected_std = 1.0  # reset after downsampling

            self.stages.append(nn.Sequential(*blocks))

        # ── Head ─────────────────────────────────────────────────────────────
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head_fc = nn.Linear(widths[-1], self.embed_dim)
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """NFNet forward: stem → 4 stages → pool → head.

        Returns:
            BackboneOutput with pooled_output [B, 1536] and 4 feature maps.
        """
        x = self.stem(x)  # [B, 128, H/4, W/4]

        feature_maps = []
        for stage in self.stages:
            x = stage(x)
            feature_maps.append(x)

        pooled = self.pool(feature_maps[-1]).flatten(1)  # [B, 1536]
        pooled = F.gelu(pooled)
        pooled = self.dropout(self.head_fc(pooled))

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
