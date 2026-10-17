"""``InternImageBackbone``: InternImage hierarchical backbone using DCNv3 operators."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.cnn.internimage.intern_image_block import InternImageBlock
from hale_core.nn.vision.cnn.internimage.intern_image_downsample import InternImageDownsample


class InternImageBackbone(BaseVisionBackbone):
    """InternImage hierarchical backbone using DCNv3 operators.

    4 stages, each producing a feature map at a different resolution.
    Adaptive receptive fields via deformable convolution.

    Args:
        stem_channels: Initial channel count (C in the paper).
        depths:        Number of InternImageBlock per stage.
        groups:        DCNv3 deformable groups per stage.
        drop_path_rate: Max stochastic depth rate.
        img_size, in_chans, drop_rate: See BaseVisionBackbone.
    """

    arch_name: str = "internimage"
    embed_dim: int = 0
    feature_dims: list[int] = []

    def __init__(
        self,
        stem_channels: int,
        depths: list[int],
        groups: list[int],
        drop_path_rate: float = 0.1,
        img_size: int = 224,
        in_chans: int = 3,
        drop_rate: float = 0.0,
        out_indices: list[int] | None = None,
    ) -> None:
        assert len(depths) == len(groups) == 4, "Need exactly 4 stages"

        # Channel progression: double each stage
        channels = [stem_channels * (2**i) for i in range(4)]
        self.feature_dims = channels
        self.embed_dim = channels[-1]

        super().__init__(
            img_size=img_size, in_chans=in_chans, drop_rate=drop_rate, out_indices=out_indices
        )

        # ── Stem: patch-like embedding with 4×4 stride-4 conv ─────────────────
        # Conv and LayerNorm are separate modules because the LN runs channels-last
        # (see forward). There is deliberately no other stem module: unused parameters
        # never get a gradient and make DistributedDataParallel fail.
        self.stem_conv = nn.Conv2d(in_chans, channels[0], kernel_size=4, stride=4, bias=False)
        self.stem_norm = nn.LayerNorm(channels[0], eps=1e-6)

        # ── Stages ───────────────────────────────────────────────────────────
        total_blocks = sum(depths)
        block_idx = 0

        self.stages: nn.ModuleList = nn.ModuleList()
        self.downsamples: nn.ModuleList = nn.ModuleList()

        for i, (n, g) in enumerate(zip(depths, groups)):
            blocks = nn.ModuleList()
            for _ in range(n):
                dp = drop_path_rate * block_idx / max(total_blocks - 1, 1)
                blocks.append(
                    InternImageBlock(
                        channels=channels[i],
                        groups=g,
                        kernel_size=3,
                        ffn_expand=4,
                        drop=0.0,
                        drop_path=dp,
                    )
                )
                block_idx += 1
            self.stages.append(blocks)
            # Downsample between stages (not after last stage)
            if i < 3:
                self.downsamples.append(InternImageDownsample(channels[i], channels[i + 1]))

        # ── Global pooling ────────────────────────────────────────────────────
        self.norm_last = nn.LayerNorm(channels[-1], eps=1e-6)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_drop = nn.Dropout(drop_rate)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """Forward pass of :class:`InternImageBackbone`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``BackboneOutput``.
        """
        # Stem
        x = self.stem_conv(x)  # [B, C, H/4, W/4]
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # channels-last for LN
        x = self.stem_norm(x).permute(0, 3, 1, 2)  # back to [B, C, H, W]

        feature_maps: list[torch.Tensor] = []

        for i, stage_blocks in enumerate(self.stages):
            for block in stage_blocks:
                x = block(x)
            feature_maps.append(x)

            if i < len(self.downsamples):
                x = self.downsamples[i](x)

        # Normalise final features for pooling
        B, C, H, W = feature_maps[-1].shape
        x_last = feature_maps[-1].permute(0, 2, 3, 1)
        x_last = self.norm_last(x_last).permute(0, 3, 1, 2)

        # Global pooled embedding
        pooled = self.pool(x_last).flatten(1)
        pooled = self.head_drop(pooled)

        # Select requested scales
        indices = self.out_indices if self.out_indices is not None else list(range(4))
        selected = [feature_maps[i] for i in indices if i < len(feature_maps)]

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=None,
            feature_maps=selected,
        )

    def freeze_up_to_stage(self, stage: int) -> None:
        """Freeze stem and stages 0..stage."""
        for p in self.stem_conv.parameters():
            p.requires_grad_(False)
        for p in self.stem_norm.parameters():
            p.requires_grad_(False)
        for i, s in enumerate(self.stages):
            if i <= stage:
                for p in s.parameters():
                    p.requires_grad_(False)
