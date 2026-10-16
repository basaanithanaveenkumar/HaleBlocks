"""``ResNeXt``: Generic ResNeXt backbone."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.cnn.resnext.res_ne_xt_bottleneck import ResNeXtBottleneck


class ResNeXt(BaseVisionBackbone):
    """Generic ResNeXt backbone.

    Same 4-stage structure as ResNet but using grouped bottleneck blocks.

    Args:
        layers:      Number of blocks per stage [s0, s1, s2, s3].
        cardinality: Transformation groups (C in the paper).
        base_width:  Width per group at stage channels (D in the paper).
        img_size:    Input resolution.
        in_chans:    Input channels.
        drop_rate:   Dropout after global pooling.
    """

    embed_dim = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(
        self,
        layers: list[int],
        cardinality: int = 32,
        base_width: int = 4,
        img_size: int = 224,
        in_chans: int = 3,
        drop_rate: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(img_size=img_size, in_chans=in_chans, drop_rate=drop_rate)

        self._cardinality = cardinality
        self._base_width = base_width
        self._in_planes = 64

        # Stem: same as ResNet — 7×7 conv → BN → ReLU → MaxPool
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.stage0 = self._make_stage(64, layers[0], stride=1)
        self.stage1 = self._make_stage(128, layers[1], stride=2)
        self.stage2 = self._make_stage(256, layers[2], stride=2)
        self.stage3 = self._make_stage(512, layers[3], stride=2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        self._stages = [self.stage0, self.stage1, self.stage2, self.stage3]
        self._init_weights()

    def _make_stage(self, planes: int, n_blocks: int, stride: int) -> nn.Sequential:
        out_ch = planes * ResNeXtBottleneck.expansion
        downsample = None
        if stride != 1 or self._in_planes != out_ch:
            downsample = nn.Sequential(
                nn.Conv2d(self._in_planes, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        blocks = [
            ResNeXtBottleneck(
                self._in_planes,
                planes,
                stride,
                cardinality=self._cardinality,
                base_width=self._base_width,
                downsample=downsample,
            )
        ]
        self._in_planes = out_ch

        for _ in range(1, n_blocks):
            blocks.append(
                ResNeXtBottleneck(
                    self._in_planes,
                    planes,
                    cardinality=self._cardinality,
                    base_width=self._base_width,
                )
            )
        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Zero-init last BN in each bottleneck for training stability
        for m in self.modules():
            if isinstance(m, ResNeXtBottleneck):
                nn.init.zeros_(m.bn3.weight)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        x = self.stem(x)

        s0 = self.stage0(x)  # [B, 256,  H/4,  W/4]
        s1 = self.stage1(s0)  # [B, 512,  H/8,  W/8]
        s2 = self.stage2(s1)  # [B, 1024, H/16, W/16]
        s3 = self.stage3(s2)  # [B, 2048, H/32, W/32]

        pooled = self.dropout(self.pool(s3).flatten(1))  # [B, 2048]

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
        if stage < 0 or stage > 3:
            raise ValueError(f"stage must be in [0, 3], got {stage}")
        for p in self.stem.parameters():
            p.requires_grad_(False)
        for i in range(stage + 1):
            for p in self._stages[i].parameters():
                p.requires_grad_(False)
