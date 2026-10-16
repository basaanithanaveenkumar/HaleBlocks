"""``ResNet``: Generic ResNet backbone."""

from __future__ import annotations

from typing import Type, Union

import torch
import torch.nn as nn

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.cnn.resnet.basic_block import BasicBlock
from hale_core.nn.vision.cnn.resnet.bottleneck import Bottleneck


class ResNet(BaseVisionBackbone):
    """Generic ResNet backbone.

    Implements the full multi-stage ResNet as a dense feature extractor, returning
    4-scale feature maps for use with FPN/BEV heads and a global pooled embedding
    for VLM/action encoders.

    Args:
        block:      Block class — BasicBlock or Bottleneck.
        layers:     Number of blocks in each of the 4 stages.
        img_size:   Expected input spatial resolution (assumed square).
        in_chans:   Input image channels (default 3 for RGB).
        drop_rate:  Dropout probability applied after global pooling.
        out_indices: Which stages to include in feature_maps (default: all 4).
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: list[int],
        img_size: int = 224,
        in_chans: int = 3,
        drop_rate: float = 0.0,
        out_indices: list[int] | None = None,
    ) -> None:
        super().__init__(
            img_size=img_size, in_chans=in_chans, drop_rate=drop_rate, out_indices=out_indices
        )

        self._block = block
        self._layers = layers

        # embed_dim is the channel count of stage 3 output
        self.embed_dim = 512 * block.expansion
        # feature_dims matches channels at each of the 4 stages
        self.feature_dims = [
            64 * block.expansion if block.expansion > 1 else 64,
            128 * block.expansion if block.expansion > 1 else 128,
            256 * block.expansion if block.expansion > 1 else 256,
            512 * block.expansion if block.expansion > 1 else 512,
        ]
        # For BasicBlock (expansion=1), channels are [64, 128, 256, 512]
        # For Bottleneck (expansion=4), channels are [256, 512, 1024, 2048]
        self.feature_dims = (
            [
                64,
                128,
                256,
                512,
            ]
            if block.expansion == 1
            else [256, 512, 1024, 2048]
        )

        self._in_planes = 64  # current channel count, updated as stages are built

        # ── Stem: 7×7 conv + BN + ReLU + MaxPool ─────────────────────────────
        # Downsamples input from H × W to H/4 × W/4 before stage 0.
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ── 4 residual stages ─────────────────────────────────────────────────
        # Stage 0: stride=1 (no downsampling, maintains H/4)
        self.stage0 = self._make_stage(block, 64, layers[0], stride=1)
        # Stages 1–3: stride=2 each, progressively halving resolution
        self.stage1 = self._make_stage(block, 128, layers[1], stride=2)
        self.stage2 = self._make_stage(block, 256, layers[2], stride=2)
        self.stage3 = self._make_stage(block, 512, layers[3], stride=2)

        # ── Global average pool + optional dropout ────────────────────────────
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        # Ordered list of stage modules for freeze_up_to_stage convenience
        self._stages = [self.stage0, self.stage1, self.stage2, self.stage3]

        self._init_weights()

    def _make_stage(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        n_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Build one ResNet stage as an nn.Sequential of `n_blocks` residual blocks.

        The first block handles any spatial downsampling (stride argument) and
        any channel dimension change (via a 1×1 projection shortcut).  Subsequent
        blocks always have stride=1 and no shortcut needed.
        """
        downsample = None
        out_channels = planes * block.expansion

        # Need a projection shortcut if spatial resolution or channel count changes
        if stride != 1 or self._in_planes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self._in_planes, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        blocks = [block(self._in_planes, planes, stride, downsample)]
        self._in_planes = out_channels

        # Subsequent blocks: in_planes == out_channels, stride=1, no downsample
        for _ in range(1, n_blocks):
            blocks.append(block(self._in_planes, planes))

        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        """Kaiming normal for all conv layers; constant BN; zero last BN in blocks."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming normal: calibrates variance for ReLU nonlinearity
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Zero-init the last BN in each residual branch so residual blocks start
        # as near-identity mappings — improves gradient flow at training start
        for m in self.modules():
            if isinstance(m, Bottleneck) and m.bn3 is not None:
                nn.init.zeros_(m.bn3.weight)
            elif isinstance(m, BasicBlock) and m.bn2 is not None:
                nn.init.zeros_(m.bn2.weight)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """Run ResNet forward pass and return multi-scale features + pooled embedding.

        Args:
            x: Float32 tensor [B, in_chans, H, W].

        Returns:
            BackboneOutput:
              pooled_output — [B, embed_dim] global representation
              patch_tokens  — None (CNNs produce spatial maps, not tokens)
              feature_maps  — [stage0_out, stage1_out, stage2_out, stage3_out]
                              shapes: [B,64,H/4,W/4], [B,128,H/8,W/8],
                                      [B,256,H/16,W/16], [B,512,H/32,W/32]
                              (channels × 4 for Bottleneck models)
        """
        x = self.stem(x)  # [B, 64, H/4, W/4]

        s0 = self.stage0(x)  # [B, 64  or 256, H/4, W/4]
        s1 = self.stage1(s0)  # [B, 128 or 512, H/8, W/8]
        s2 = self.stage2(s1)  # [B, 256 or 1024, H/16, W/16]
        s3 = self.stage3(s2)  # [B, 512 or 2048, H/32, W/32]

        # Global pooled embedding: flatten spatial dims after AdaptiveAvgPool
        pooled = self.dropout(self.pool(s3).flatten(1))  # [B, embed_dim]

        # Select which scales to return
        all_maps = [s0, s1, s2, s3]
        if self.out_indices is not None:
            all_maps = [all_maps[i] for i in self.out_indices]

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=None,  # CNNs have no patch token abstraction
            feature_maps=all_maps,
            attentions=None,
        )

    # ── Stage freezing ────────────────────────────────────────────────────────

    def freeze_up_to_stage(self, stage: int) -> None:
        """Freeze the stem and stages 0 through `stage` (inclusive).

        Useful for fine-tuning: keep early low-level features fixed and only
        update later stages.

        Args:
            stage: 0-based stage index.  E.g. stage=1 freezes stem+stage0+stage1.

        Raises:
            ValueError: if stage is out of range [0, 3].
        """
        if stage < 0 or stage > 3:
            raise ValueError(f"stage must be in [0, 3], got {stage}")

        # Always freeze stem along with stage 0
        for p in self.stem.parameters():
            p.requires_grad_(False)

        for i in range(stage + 1):
            for p in self._stages[i].parameters():
                p.requires_grad_(False)
