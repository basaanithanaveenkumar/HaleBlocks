"""``MobileOneBackbone``: MobileOne hierarchical CNN backbone."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.cnn.mobile_one._helpers import _ConvBN, _make_divisible
from hale_core.nn.vision.cnn.mobile_one.mobile_one_block import MobileOneBlock


class MobileOneBackbone(BaseVisionBackbone):
    """MobileOne hierarchical CNN backbone.

    4 stages with progressively wider channels and stride-2 downsampling.
    Multi-scale feature maps returned for FPN/BEV/detection heads.

    Args:
        width_mult: Per-stage width multipliers over base channels [48, 96, 192, 512].
        num_blocks: k branches per block (1 = reparam-ready; 4 = rich training).
        img_size, in_chans, drop_rate: See BaseVisionBackbone.
    """

    arch_name: str = "mobile_one"
    embed_dim: int = 512  # overridden per variant
    feature_dims: list[int] = []

    _BASE_CHANNELS = [48, 96, 192, 512]  # stage output channels before width scaling
    _DEPTHS = [2, 8, 10, 1]  # number of blocks per stage

    def __init__(
        self,
        width_mult: list[float] = (1.0, 1.0, 1.0, 1.0),
        num_blocks: int = 1,
        img_size: int = 224,
        in_chans: int = 3,
        drop_rate: float = 0.0,
        out_indices: list[int] | None = None,
    ) -> None:
        # Compute actual channel counts
        stage_dims = [_make_divisible(c * m, 8) for c, m in zip(self._BASE_CHANNELS, width_mult)]
        self.feature_dims = stage_dims
        self.embed_dim = stage_dims[-1]

        super().__init__(
            img_size=img_size, in_chans=in_chans, drop_rate=drop_rate, out_indices=out_indices
        )

        self._num_blocks = num_blocks

        # ── Stem (stride-2 conv + MobileOneBlock) ────────────────────────────
        stem_out = _make_divisible(self._BASE_CHANNELS[0] * width_mult[0] / 2, 8)
        self.stem = nn.Sequential(
            _ConvBN(in_chans, stem_out, kernel_size=3, stride=2),
            nn.ReLU(),
            MobileOneBlock(stem_out, stage_dims[0], stride=2, num_blocks=num_blocks),
        )

        # ── Stages ───────────────────────────────────────────────────────────
        # Stage 0 is produced by stem; stages 1-3 further downsample
        self.stages: nn.ModuleList = nn.ModuleList()
        in_c = stage_dims[0]
        for i, (out_c, n_depth) in enumerate(zip(stage_dims[1:], self._DEPTHS[1:]), start=1):
            blocks: list[nn.Module] = []
            # First block: stride 2 to halve resolution
            blocks.append(MobileOneBlock(in_c, out_c, stride=2, num_blocks=num_blocks))
            # Remaining blocks: same resolution
            for _ in range(n_depth - 1):
                blocks.append(MobileOneBlock(out_c, out_c, stride=1, num_blocks=num_blocks))
            self.stages.append(nn.Sequential(*blocks))
            in_c = out_c

        # ── Global pooling for classification/embedding ───────────────────────
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_drop = nn.Dropout(drop_rate)

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
        # Stage 0 output from stem
        x = self.stem(x)
        all_features: list[torch.Tensor] = [x]

        for stage in self.stages:
            x = stage(x)
            all_features.append(x)

        # Select requested stages
        indices = self.out_indices if self.out_indices is not None else list(range(4))
        feature_maps = [all_features[i] for i in indices if i < len(all_features)]

        # Global pooled embedding
        pooled = self.pool(all_features[-1]).flatten(1)
        pooled = self.head_drop(pooled)

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=None,
            feature_maps=feature_maps,
        )

    def reparametrize(self) -> None:
        """Fuse all multi-branch blocks for inference (run once before deployment)."""
        for m in self.modules():
            if isinstance(m, MobileOneBlock):
                m.reparametrize()

    def freeze_up_to_stage(self, stage: int) -> None:
        """Freeze stem and the first `stage` stages (0-indexed)."""
        for p in self.stem.parameters():
            p.requires_grad_(False)
        for i, s in enumerate(self.stages):
            if i < stage:
                for p in s.parameters():
                    p.requires_grad_(False)
