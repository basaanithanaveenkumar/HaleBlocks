"""``EfficientNetV2Backbone``: EfficientNetV2 backbone returning multi-scale feature maps."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.cnn.efficientnet._helpers import _StageConfig
from hale_core.nn.vision.cnn.efficientnet.fused_mb_conv import FusedMBConv
from hale_core.nn.vision.cnn.efficientnet.mb_conv import MBConv


class EfficientNetV2Backbone(BaseVisionBackbone):
    """EfficientNetV2 backbone returning multi-scale feature maps.

    Multi-scale output:
      feature_maps[0]  stride-4   (after stem)
      feature_maps[1]  stride-8   (after stage 1)
      feature_maps[2]  stride-16  (after stages 2+3)
      feature_maps[3]  stride-32  (after stages 4+5)

    Args:
        stages:          List of _StageConfig (defines the full architecture).
        head_channels:   Output channels after the head 1×1 conv.
        drop_path_rate:  Max stochastic depth rate (linearly scaled per block).
        img_size, in_chans, drop_rate: See BaseVisionBackbone.
    """

    arch_name: str = "efficientnet_v2"
    embed_dim: int = 1280
    feature_dims: list[int] = []

    def __init__(
        self,
        stages: list[_StageConfig],
        head_channels: int = 1280,
        drop_path_rate: float = 0.0,
        img_size: int = 224,
        in_chans: int = 3,
        drop_rate: float = 0.0,
        out_indices: list[int] | None = None,
    ) -> None:
        self.embed_dim = head_channels

        # Feature dims at strides 4, 8, 16, 32 — to be filled below
        super().__init__(
            img_size=img_size, in_chans=in_chans, drop_rate=drop_rate, out_indices=out_indices
        )

        # Count total blocks for stochastic depth schedule
        total_blocks = sum(s.num_blocks for s in stages)
        block_idx = 0

        # ── Stem (stride-2, produces H/2) ────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, stages[0].in_chs, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stages[0].in_chs),
            nn.SiLU(inplace=True),
        )

        # ── Stages ───────────────────────────────────────────────────────────
        # We group stages by their STRIDE to build a 4-scale feature pyramid.
        # stride-2 transitions mark new pyramid levels.
        self.stage_groups: nn.ModuleList = nn.ModuleList()
        current_stride = 2

        groups: list[list[nn.Module]] = []
        current_group: list[nn.Module] = []
        last_out_chs = stages[0].in_chs

        for s in stages:
            # dp_rate linearly increases from 0 to drop_path_rate
            for i in range(s.num_blocks):
                dp = drop_path_rate * block_idx / max(total_blocks - 1, 1)
                stride_i = s.stride if i == 0 else 1  # stride only on first block

                if stride_i == 2 and current_group:
                    # Start a new group at each downsample
                    groups.append(current_group)
                    current_group = []
                    current_stride *= 2

                if s.block_type == "fused":
                    blk = FusedMBConv(
                        last_out_chs,
                        s.out_chs,
                        stride=stride_i,
                        expand_ratio=s.expand_ratio,
                        se_ratio=s.se_ratio,
                        drop_path=dp,
                    )
                else:
                    blk = MBConv(
                        last_out_chs,
                        s.out_chs,
                        stride=stride_i,
                        expand_ratio=s.expand_ratio,
                        kernel_size=s.kernel_size,
                        se_ratio=s.se_ratio,
                        drop_path=dp,
                    )
                current_group.append(blk)
                last_out_chs = s.out_chs
                block_idx += 1

        if current_group:
            groups.append(current_group)

        self.stage_groups = nn.ModuleList(nn.Sequential(*g) for g in groups)

        # ── Head conv (H/32 resolution, embed_dim channels) ──────────────────
        self.head_conv = nn.Sequential(
            nn.Conv2d(last_out_chs, head_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_channels),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_drop = nn.Dropout(drop_rate)

        # Feature dims are the output channels of each stride group
        # (roughly strides 4, 8, 16, 32 — may vary by stage config)
        self.feature_dims = [
            g[-1][-1]
            if hasattr(g[-1], "__len__")
            else (list(g[-1].parameters())[0].shape[0] if list(g[-1].parameters()) else 0)
            for g in groups[:4]
        ]
        # Simpler: just use the last layer's out_channels per group
        self.feature_dims = self._compute_feature_dims(groups)

        self._init_weights()

    def _compute_feature_dims(self, groups) -> list[int]:
        """Get output channel count of each stage group."""
        dims = []
        for g in groups:
            last_block = g[-1]
            # Walk through Sequential to find the last Conv2d or BN
            for m in reversed(list(last_block.modules())):
                if isinstance(m, nn.BatchNorm2d):
                    dims.append(m.num_features)
                    break
        return dims[:4]

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """Forward pass of :class:`EfficientNetV2Backbone`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``BackboneOutput``.
        """
        x = self.stem(x)
        feature_maps: list[torch.Tensor] = []

        for group in self.stage_groups:
            x = group(x)
            feature_maps.append(x)

        # Apply head conv at final scale
        x = self.head_conv(x)
        # Replace last feature map with head-conv output for richer features
        if feature_maps:
            feature_maps[-1] = x

        # Select requested output scales
        indices = (
            self.out_indices if self.out_indices is not None else list(range(len(feature_maps)))
        )
        selected = [feature_maps[i] for i in indices if i < len(feature_maps)]

        pooled = self.pool(x).flatten(1)
        pooled = self.head_drop(pooled)

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=None,
            feature_maps=selected,
        )

    def freeze_up_to_stage(self, stage: int) -> None:
        """Freeze stem and stage groups 0..stage."""
        for p in self.stem.parameters():
            p.requires_grad_(False)
        for i, g in enumerate(self.stage_groups):
            if i <= stage:
                for p in g.parameters():
                    p.requires_grad_(False)
