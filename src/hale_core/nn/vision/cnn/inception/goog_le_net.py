"""``GoogLeNet``: GoogLeNet (Inception-v1) backbone."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.cnn.inception.conv_bn_relu import ConvBnRelu
from hale_core.nn.vision.cnn.inception.inception_module_v1 import InceptionModuleV1


class GoogLeNet(BaseVisionBackbone):
    """GoogLeNet (Inception-v1) backbone.

    Returns 4-scale feature maps (stem + one per MaxPool boundary) plus a
    1024-dim global pooled embedding from the final Inception stage.
    """

    arch_name = "inception_v1"
    embed_dim = 1024
    feature_dims = [192, 480, 832, 1024]

    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 3,
        drop_rate: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(img_size=img_size, in_chans=in_chans, drop_rate=drop_rate)

        # Stem: aggressive spatial reduction before Inception modules begin
        self.stem = nn.Sequential(
            ConvBnRelu(in_chans, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ConvBnRelu(64, 64, kernel_size=1),
            ConvBnRelu(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )  # → [B, 192, H/8, W/8]

        # Stage 1: Inception 3a (256 ch) + 3b (480 ch) → MaxPool → [B, 480, H/16]
        self.stage1 = nn.Sequential(
            InceptionModuleV1(192, 64, 96, 128, 16, 32, 32),  # 3a → 256
            InceptionModuleV1(256, 128, 128, 192, 32, 96, 64),  # 3b → 480
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Stage 2: Inception 4a–4e (up to 832 ch) → MaxPool → [B, 832, H/32]
        self.stage2 = nn.Sequential(
            InceptionModuleV1(480, 192, 96, 208, 16, 48, 64),  # 4a → 512
            InceptionModuleV1(512, 160, 112, 224, 24, 64, 64),  # 4b → 512
            InceptionModuleV1(512, 128, 128, 256, 24, 64, 64),  # 4c → 512
            InceptionModuleV1(512, 112, 144, 288, 32, 64, 64),  # 4d → 528
            InceptionModuleV1(528, 256, 160, 320, 32, 128, 128),  # 4e → 832
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Stage 3: Inception 5a (832 ch) + 5b (1024 ch) → GlobalAvgPool
        self.stage3 = nn.Sequential(
            InceptionModuleV1(832, 256, 160, 320, 32, 128, 128),  # 5a → 832
            InceptionModuleV1(832, 384, 192, 384, 48, 128, 128),  # 5b → 1024
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
        """Forward pass of :class:`GoogLeNet`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``BackboneOutput``.
        """
        s0 = self.stem(x)  # [B, 192, H/8, W/8]
        s1 = self.stage1(s0)  # [B, 480, H/16, W/16]
        s2 = self.stage2(s1)  # [B, 832, H/32, W/32]
        s3 = self.stage3(s2)  # [B, 1024, H/32, W/32]

        pooled = self.dropout(self.pool(s3).flatten(1))  # [B, 1024]

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
