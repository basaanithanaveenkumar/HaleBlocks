"""``MobileNetV1``: MobileNetV1 backbone with width multiplier."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.cnn.mobilenet_v1._helpers import _DWSConv


class MobileNetV1(BaseVisionBackbone):
    """MobileNetV1 backbone with width multiplier.

    Returns 4-scale feature maps (one per stride-2 stage) and a global pooled
    embedding for use with VLM projectors and action encoders.

    Args:
        alpha:      Width multiplier ∈ {0.25, 0.5, 0.75, 1.0}.  Scales all
                    channel counts linearly; controls accuracy/efficiency trade-off.
        img_size:   Expected input spatial resolution.
        in_chans:   Input image channels.
        drop_rate:  Dropout after global pool.
        out_indices: Which stages to include in feature_maps.
    """

    arch_name: str = "mobilenet_v1"
    embed_dim: int = 1024
    feature_dims: list[int] = [64, 128, 256, 512, 1024]

    def __init__(
        self,
        alpha: float = 1.0,
        img_size: int = 224,
        in_chans: int = 3,
        drop_rate: float = 0.0,
        out_indices: list[int] | None = None,
    ) -> None:
        super().__init__(
            img_size=img_size, in_chans=in_chans, drop_rate=drop_rate, out_indices=out_indices
        )

        def _c(n: int) -> int:
            """Scale channel count by alpha, ensure minimum 1."""
            return max(1, int(n * alpha))

        self.embed_dim = _c(1024)
        self.feature_dims = [_c(64), _c(128), _c(256), _c(512), _c(1024)]

        # ── Stem: standard Conv + first DWSConv ───────────────────────────────
        # One standard conv at input (depthwise on 3-channel input is odd),
        # then immediately a DWSConv to the first stage output channels.
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, _c(32), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(_c(32)),
            nn.ReLU(inplace=True),
            _DWSConv(_c(32), _c(64), stride=1),
        )

        # ── Stages — each starts with a stride-2 DWSConv then stride-1 blocks ─
        # Stage 0: 64→128 (s2) → 128→128 (s1)
        self.stage0 = nn.Sequential(
            _DWSConv(_c(64), _c(128), stride=2),
            _DWSConv(_c(128), _c(128), stride=1),
        )

        # Stage 1: 128→256 (s2) → 256→256 (s1)
        self.stage1 = nn.Sequential(
            _DWSConv(_c(128), _c(256), stride=2),
            _DWSConv(_c(256), _c(256), stride=1),
        )

        # Stage 2: 256→512 (s2) → 5×512→512 (s1) — most compute here
        self.stage2 = nn.Sequential(
            _DWSConv(_c(256), _c(512), stride=2),
            *[_DWSConv(_c(512), _c(512), stride=1) for _ in range(5)],
        )

        # Stage 3: 512→1024 (s2) → 1024→1024 (s1)
        self.stage3 = nn.Sequential(
            _DWSConv(_c(512), _c(1024), stride=2),
            _DWSConv(_c(1024), _c(1024), stride=1),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        self._stages = [self.stage0, self.stage1, self.stage2, self.stage3]
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
        """Run MobileNetV1 forward pass.

        Args:
            x: [B, in_chans, H, W]

        Returns:
            BackboneOutput:
              pooled_output — [B, embed_dim]
              feature_maps  — [s0, s1, s2, s3] at strides [4, 8, 16, 32]
        """
        x = self.stem(x)  # [B, _c(64), H/2, W/2]

        s0 = self.stage0(x)  # [B, _c(128), H/4, W/4]
        s1 = self.stage1(s0)  # [B, _c(256), H/8, W/8]
        s2 = self.stage2(s1)  # [B, _c(512), H/16, W/16]
        s3 = self.stage3(s2)  # [B, _c(1024), H/32, W/32]

        pooled = self.dropout(self.pool(s3).flatten(1))

        all_maps = [s0, s1, s2, s3]
        if self.out_indices is not None:
            all_maps = [all_maps[i] for i in self.out_indices]

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=None,
            feature_maps=all_maps,
            attentions=None,
        )

    def freeze_up_to_stage(self, stage: int) -> None:
        """Freeze stem and stages 0..stage (inclusive).

        Args:
            stage: 0-based index, must be in [0, 3].
        """
        if stage < 0 or stage > 3:
            raise ValueError(f"stage must be in [0, 3], got {stage}")
        for p in self.stem.parameters():
            p.requires_grad_(False)
        for i in range(stage + 1):
            for p in self._stages[i].parameters():
                p.requires_grad_(False)
