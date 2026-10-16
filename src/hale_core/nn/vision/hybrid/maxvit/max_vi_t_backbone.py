"""``MaxViTBackbone``: MaxViT vision backbone."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.hybrid.maxvit._helpers import _trunc_normal_
from hale_core.nn.vision.hybrid.maxvit.max_vi_t_stage import MaxViTStage


class MaxViTBackbone(BaseVisionBackbone):
    """MaxViT vision backbone.

    4-stage hierarchy. Each stage: strided MBConv + N MaxViT blocks.
    Block attention window is block_size×block_size (default 8).
    Grid attention uses the same window size as dilated stride.

    Args:
        dim:         Base channel count (stage 0 has this many channels).
        depths:      Number of blocks per stage [d0, d1, d2, d3].
        block_size:  Local attention window size (must divide feature map size).
        drop_path_rate: Total stochastic depth drop rate.
        **base_kwargs: Passed to BaseVisionBackbone.
    """

    def __init__(
        self,
        dim: int = 64,
        depths: list = None,
        block_size: int = 8,
        drop_path_rate: float = 0.2,
        **base_kwargs,
    ) -> None:
        super().__init__(**base_kwargs)
        depths = depths or [2, 2, 5, 2]
        stage_dims = [dim * (2**i) for i in range(len(depths))]

        # Class-level attributes
        self.__class__.feature_dims = stage_dims
        self.__class__.embed_dim = stage_dims[-1]

        # Distribute stochastic depth rates
        total = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total)]
        idx = 0

        def _take(n: int) -> list:
            nonlocal idx
            r = dp_rates[idx : idx + n]
            idx += n
            return r

        # Stem: two 3×3 convs (stride-2 + stride-1)  input H → H/2
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_chans, dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU(),
        )

        # 4 stages: each halves H and doubles C
        # Stage input channels: [dim, dim, dim*2, dim*4]  (stem produces dim, stage 0 doubles)
        stem_ch = dim
        self.stages = nn.ModuleList()
        in_ch = stem_ch
        for i, (d, out_ch) in enumerate(zip(depths, stage_dims)):
            heads = max(1, out_ch // 32)  # ~1 head per 32 channels
            self.stages.append(
                MaxViTStage(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    depth=d,
                    num_heads=heads,
                    block_size=block_size,
                    drop_path_rates=_take(d),
                )
            )
            in_ch = out_ch

        self.norm = nn.BatchNorm2d(stage_dims[-1])
        self.drop = nn.Dropout(self.drop_rate)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                _trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def freeze_up_to_stage(self, stage: int) -> None:
        """Freeze stem + stages 0..stage (inclusive)."""
        for p in self.stem.parameters():
            p.requires_grad_(False)
        for i in range(min(stage + 1, len(self.stages))):
            for p in self.stages[i].parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            BackboneOutput with 4 feature maps and pooled_output
        """
        x = self.stem(x)  # [B, dim, H/2, W/2]

        feature_maps = []
        for stage in self.stages:
            x = stage(x)  # Each stage halves H,W and doubles C
            feature_maps.append(x)

        x = self.norm(x)
        pooled = x.mean(dim=(2, 3))  # [B, stage_dims[-1]]
        pooled = self.drop(pooled)

        if self.out_indices is not None:
            feature_maps = [feature_maps[i] for i in self.out_indices]

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=None,
            feature_maps=feature_maps,
        )
