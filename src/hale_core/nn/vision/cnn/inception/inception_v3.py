"""``InceptionV3``: Inception-v3 backbone (no auxiliary classifiers)."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.cnn.inception._inception_type_a import _InceptionTypeA
from hale_core.nn.vision.cnn.inception._inception_type_b import _InceptionTypeB
from hale_core.nn.vision.cnn.inception._inception_type_c import _InceptionTypeC
from hale_core.nn.vision.cnn.inception._reduction_a import _ReductionA
from hale_core.nn.vision.cnn.inception._reduction_b import _ReductionB
from hale_core.nn.vision.cnn.inception.conv_bn_relu import ConvBnRelu


class InceptionV3(BaseVisionBackbone):
    """Inception-v3 backbone (no auxiliary classifiers).

    Factorised convolutions throughout; embed_dim=2048 from mixed_7c.
    """

    arch_name = "inception_v3"
    embed_dim = 2048
    feature_dims = [192, 768, 1280, 2048]

    def __init__(
        self,
        img_size: int = 299,
        in_chans: int = 3,
        drop_rate: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(img_size=img_size, in_chans=in_chans, drop_rate=drop_rate)

        # Stem: 5-step conv sequence → [B, 192, H/8, W/8] approx
        self.stem = nn.Sequential(
            ConvBnRelu(in_chans, 32, kernel_size=3, stride=2),
            ConvBnRelu(32, 32, kernel_size=3),
            ConvBnRelu(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
            ConvBnRelu(64, 80, kernel_size=1),
            ConvBnRelu(80, 192, kernel_size=3),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Stage 1 — Type-A × 3: 192 → 256 → 288 → 288
        # _InceptionTypeA outputs 64+64+96+pool = 64+64+96+32 = 256 (pool=32)
        # and 64+64+96+64 = 288 (pool=64)
        self.stage1 = nn.Sequential(
            _InceptionTypeA(192, pool_features=32),  # → 256
            _InceptionTypeA(256, pool_features=64),  # → 288
            _InceptionTypeA(288, pool_features=64),  # → 288
        )

        # Reduction A: 288 → 768
        self.reduction_a = _ReductionA(in_ch=288)

        # Stage 2 — Type-B × 4: 768 → 768 (channels fixed at 192*4=768)
        self.stage2 = nn.Sequential(
            _InceptionTypeB(768, channels_7x7=128),  # → 768
            _InceptionTypeB(768, channels_7x7=160),  # → 768
            _InceptionTypeB(768, channels_7x7=160),  # → 768
            _InceptionTypeB(768, channels_7x7=192),  # → 768
        )

        # Reduction B: 768 → 1280
        self.reduction_b = _ReductionB(in_ch=768)

        # Stage 3 — Type-C × 2: 1280 → 2048 → 2048
        self.stage3 = nn.Sequential(
            _InceptionTypeC(in_ch=1280),  # → 2048
            _InceptionTypeC(in_ch=2048),  # → 2048
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        s0 = self.stem(x)  # [B, 192, ~H/32, ~W/32]
        s1 = self.reduction_a(self.stage1(s0))  # [B, 768, ...]
        s2 = self.reduction_b(self.stage2(s1))  # [B, 1280, ...]
        s3 = self.stage3(s2)  # [B, 2048, ...]

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
        stages = [self.stem, self.stage1, self.stage2, self.stage3]
        for i in range(stage + 1):
            for p in stages[i].parameters():
                p.requires_grad_(False)
