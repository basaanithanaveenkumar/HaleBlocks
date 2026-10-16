"""``CoAtNetBackbone``: CoAtNet vision backbone."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.hybrid.coatnet._helpers import _trunc_normal_
from hale_core.nn.vision.hybrid.coatnet.co_at_net_attn_stage import CoAtNetAttnStage
from hale_core.nn.vision.hybrid.coatnet.co_at_net_conv_stage import CoAtNetConvStage


class CoAtNetBackbone(BaseVisionBackbone):
    """CoAtNet vision backbone.

    5-stage architecture: Conv Stem + 2 MBConv stages + 2 Attention stages.

    Args:
        dims:    Channel counts for [stem, S1, S2, S3, S4].
        depths:  Block counts for [stem, S1, S2, S3, S4].
                 Stem depth is number of conv layers (typically 2).
        num_heads_attn: Attention heads for S3 and S4.
        drop_path_rate: Total stochastic depth rate.
        **base_kwargs: Passed to BaseVisionBackbone.
    """

    def __init__(
        self,
        dims: list = None,
        depths: list = None,
        num_heads_attn: int = 8,
        drop_path_rate: float = 0.2,
        **base_kwargs,
    ) -> None:
        super().__init__(**base_kwargs)
        dims = dims or [64, 96, 192, 384, 768]
        depths = depths or [2, 3, 5, 2, 2]

        # Class-level attributes for base __repr__
        self.__class__.feature_dims = dims[1:]  # S1..S4 channel counts
        self.__class__.embed_dim = dims[-1]

        # Distribute drop_path evenly over all non-stem blocks
        total = sum(depths[1:])
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total)]
        idx = 0

        # S0: Conv stem — two stride-2 convolutions  (input H → H/4)
        self.stem = nn.Sequential(
            nn.Conv2d(self.in_chans, dims[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
            nn.Conv2d(dims[0], dims[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.GELU(),
        )

        def _take(n: int) -> list:
            nonlocal idx
            r = dp_rates[idx : idx + n]
            idx += n
            return r

        # S1: MBConv blocks, stride-2 → H/8
        self.s1 = CoAtNetConvStage(
            dims[0], dims[1], depths[1], stride=2, drop_path_rates=_take(depths[1])
        )
        # S2: MBConv blocks, stride-2 → H/16
        self.s2 = CoAtNetConvStage(
            dims[1], dims[2], depths[2], stride=2, drop_path_rates=_take(depths[2])
        )
        # S3: Attention blocks, stride-2 → H/32
        heads3 = max(1, dims[3] // 64)  # ~1 head per 64 dim
        self.s3 = CoAtNetAttnStage(
            dims[2],
            dims[3],
            depths[3],
            num_heads=heads3,
            stride=2,
            drop_path_rates=_take(depths[3]),
        )
        # S4: Attention blocks, stride-1 (maintain resolution)
        heads4 = max(1, dims[4] // 64)
        self.s4 = CoAtNetAttnStage(
            dims[3],
            dims[4],
            depths[4],
            num_heads=heads4,
            stride=1,
            drop_path_rates=_take(depths[4]),
        )

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
        """Freeze stem (stage -1) + conv/attn stages 0..stage."""
        all_stages = [self.stem, self.s1, self.s2, self.s3, self.s4]
        for i in range(min(stage + 2, len(all_stages))):  # +2: stem counts as stage -1
            for p in all_stages[i].parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            BackboneOutput with 4 feature maps (S1..S4) and pooled_output
        """
        x = self.stem(x)  # [B, dims[0], H/4, W/4]

        s1 = self.s1(x)  # [B, dims[1], H/8,  W/8]
        s2 = self.s2(s1)  # [B, dims[2], H/16, W/16]
        s3 = self.s3(s2)  # [B, dims[3], H/32, W/32]
        s4 = self.s4(s3)  # [B, dims[4], H/32, W/32] (stride-1)

        # Global average pool over final spatial map for pooled embedding
        pooled = s4.mean(dim=(2, 3))  # [B, dims[4]]
        pooled = self.drop(pooled)

        feature_maps = [s1, s2, s3, s4]
        if self.out_indices is not None:
            feature_maps = [feature_maps[i] for i in self.out_indices]

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=None,
            feature_maps=feature_maps,
        )
