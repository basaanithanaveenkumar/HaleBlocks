"""``RegNet``: RegNet: principled network from quantised design space."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.cnn.regnet.res_bottleneck_block import ResBottleneckBlock


class RegNet(BaseVisionBackbone):
    """RegNet: principled network from quantised design space.

    Args:
        depths:      Number of blocks in each of the 4 stages.
        widths:      Channel width at each of the 4 stages (must be divisible by group_width).
        group_width: Shared group width across all stages.
        use_se:      If True, add Squeeze-Excite (RegNetY); else RegNetX.
        img_size:    Input resolution.
        in_chans:    Input channels.
        drop_rate:   Dropout after global pool.
        out_indices: Which stage feature maps to return.
    """

    def __init__(
        self,
        depths: list[int],
        widths: list[int],
        group_width: int,
        use_se: bool = False,
        img_size: int = 224,
        in_chans: int = 3,
        drop_rate: float = 0.0,
        out_indices: list[int] | None = None,
    ) -> None:
        super().__init__(
            img_size=img_size, in_chans=in_chans, drop_rate=drop_rate, out_indices=out_indices
        )

        self.embed_dim = widths[-1]
        self.feature_dims = list(widths)

        # Stem: Conv(3→32, 3×3, s2) + BN + ReLU
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 4 stages with progressive downsampling
        self.stages = nn.ModuleList()
        in_ch = 32
        for stage_idx, (depth, width) in enumerate(zip(depths, widths)):
            blocks = []
            for i in range(depth):
                stride = 2 if (i == 0 and stage_idx > 0) else 1
                blocks.append(
                    ResBottleneckBlock(
                        in_channels=in_ch,
                        out_channels=width,
                        stride=stride,
                        group_width=group_width,
                        use_se=use_se,
                    )
                )
                in_ch = width
            self.stages.append(nn.Sequential(*blocks))

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """RegNet forward: stem → 4 stages → global pool.

        Returns:
            BackboneOutput with pooled_output [B, widths[-1]] and 4 feature maps.
        """
        x = self.stem(x)  # [B, 32, H/2, W/2]

        feature_maps = []
        for stage in self.stages:
            x = stage(x)
            feature_maps.append(x)

        pooled = self.dropout(self.pool(feature_maps[-1]).flatten(1))

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
