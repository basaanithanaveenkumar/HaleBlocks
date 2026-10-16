"""``EfficientNetB``: Original EfficientNet B0-B7 backbone."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.cnn.efficientnet_b._helpers import _B0_STAGES, _make_divisible
from hale_core.nn.vision.cnn.efficientnet_b.mb_conv_b import MBConvB


class EfficientNetB(BaseVisionBackbone):
    """Original EfficientNet B0-B7 backbone.

    Scales the B0 baseline architecture using compound depth/width multipliers.
    Returns a 4-scale feature pyramid compatible with FPN/BEV heads.

    The feature pyramid groups stages by their spatial stride:
      feature_maps[0] → stride 4  (stem output)
      feature_maps[1] → stride 8  (after first stride-2 stage)
      feature_maps[2] → stride 16 (after second stride-2 stage)
      feature_maps[3] → stride 32 (after third stride-2 stage + remaining stages + head)

    Args:
        depth_mult:   Depth scaling factor (multiplies num_blocks per stage).
        width_mult:   Width scaling factor (multiplies channel counts).
        head_chs:     Head 1×1 conv output channels (default 1280).
        drop_path_rate: Max stochastic depth rate.
        dropout_rate: Classifier dropout (applied to pooled embedding).
        img_size, in_chans, drop_rate, out_indices: See BaseVisionBackbone.
    """

    arch_name: str = "efficientnet_b"
    embed_dim: int = 1280
    feature_dims: list[int] = []

    def __init__(
        self,
        depth_mult: float = 1.0,
        width_mult: float = 1.0,
        head_chs: int = 1280,
        drop_path_rate: float = 0.2,
        img_size: int = 224,
        in_chans: int = 3,
        drop_rate: float = 0.2,
        out_indices: list[int] | None = None,
    ) -> None:
        self.embed_dim = head_chs
        super().__init__(
            img_size=img_size, in_chans=in_chans, drop_rate=drop_rate, out_indices=out_indices
        )

        def _w(c: int) -> int:
            return _make_divisible(c * width_mult, 8)

        def _d(n: int) -> int:
            return max(1, math.ceil(n * depth_mult))

        # ── Stem: stride-2, 3→stem_chs ───────────────────────────────────────
        stem_chs = _w(32)
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, stem_chs, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_chs),
            nn.SiLU(inplace=True),
        )

        # ── Build scaled stages ───────────────────────────────────────────────
        total_blocks = sum(_d(s.num_blocks) for s in _B0_STAGES)
        block_idx = 0

        # Group blocks by feature map scale (stride-2 transitions)
        # Groups: [stem→s1], [s2], [s3], [s4+s5+s6+s7+head]
        # stride-2 stages: 2 (24ch), 3 (40ch), 4 (80ch), 6 (192ch)
        # We track stride to assign blocks to feature pyramid levels.
        stride_groups: list[list[nn.Module]] = [[] for _ in range(4)]
        current_group = 0
        stride_counts = [0]  # cumulative stride-doublings seen so far

        in_chs = stem_chs
        stride_2_seen = 0

        stage_modules: list[nn.Sequential] = []

        for s in _B0_STAGES:
            out_chs = _w(s.out_chs)
            num_blocks = _d(s.num_blocks)
            stage_blks: list[nn.Module] = []

            for i in range(num_blocks):
                dp = drop_path_rate * block_idx / max(total_blocks - 1, 1)
                blk = MBConvB(
                    in_chs=in_chs,
                    out_chs=out_chs,
                    stride=s.stride if i == 0 else 1,
                    expand_ratio=s.expand_ratio,
                    kernel_size=s.kernel_size,
                    se_ratio=0.25,
                    drop_path=dp,
                )
                stage_blks.append(blk)
                in_chs = out_chs
                block_idx += 1

            stage_modules.append(nn.Sequential(*stage_blks))
            if s.stride == 2:
                stride_2_seen += 1

        self.stages = nn.ModuleList(stage_modules)

        # ── Head conv ─────────────────────────────────────────────────────────
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_chs, head_chs, kernel_size=1, bias=False),
            nn.BatchNorm2d(head_chs),
            nn.SiLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head_drop = nn.Dropout(drop_rate)

        # Feature dims: output channels of stride-2 transition stages + head
        # Stages with stride=2: indices 1, 2, 3, 5 in _B0_STAGES (0-indexed)
        stride2_stages = [i for i, s in enumerate(_B0_STAGES) if s.stride == 2]
        # feature_dims[0] = channels after stem (stride-4)
        # feature_dims[1..3] = channels after each of the first 3 stride-2 stages
        stride2_chs = [_w(_B0_STAGES[i].out_chs) for i in stride2_stages[:3]]
        self.feature_dims = [stem_chs] + stride2_chs + [head_chs]
        # Trim to 4 levels
        self.feature_dims = self.feature_dims[:4]

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
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """Run EfficientNet-B forward pass.

        Args:
            x: [B, in_chans, H, W]

        Returns:
            BackboneOutput:
              pooled_output — [B, head_chs]
              feature_maps  — 4-scale pyramid
        """
        # Stem output: stride-4 (first feature map level)
        x = self.stem(x)
        feature_maps: list[torch.Tensor] = [x]

        stride_count = 0
        for stage in self.stages:
            x = stage(x)
            # Detect stride-2 stages by checking the first block's stride
            first_block = stage[0] if len(stage) > 0 else None
            if first_block is not None:
                # Check if first block had stride=2 (has no skip connection)
                if not getattr(first_block, "has_skip", True):
                    # This was a stride-2 block — emit feature map
                    if stride_count < 3:
                        feature_maps.append(x)
                    stride_count += 1

        # Apply head conv and use its output for the final feature scale
        x = self.head_conv(x)
        # Ensure we have exactly 4 feature maps: stem + 3 stride-2 transitions
        while len(feature_maps) < 4:
            feature_maps.append(x)
        feature_maps[-1] = x  # last map always uses head output

        indices = (
            self.out_indices if self.out_indices is not None else list(range(len(feature_maps)))
        )
        selected = [feature_maps[i] for i in indices if i < len(feature_maps)]

        pooled = self.head_drop(self.pool(x).flatten(1))

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=None,
            feature_maps=selected,
            attentions=None,
        )

    def freeze_up_to_stage(self, stage: int) -> None:
        """Freeze stem + stages 0..stage of the stage list."""
        for p in self.stem.parameters():
            p.requires_grad_(False)
        for i, s in enumerate(self.stages):
            if i <= stage:
                for p in s.parameters():
                    p.requires_grad_(False)
