"""``SEResNet``: SE-ResNet / SE-ResNeXt backbone."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.cnn.senet.se_bottleneck import SEBottleneck


class SEResNet(BaseVisionBackbone):
    """SE-ResNet / SE-ResNeXt backbone.

    Identical stage structure to ResNet-50/101/152 but uses SEBottleneck blocks.
    Returns 4-scale multi-resolution feature maps + global pooled embedding.

    Args:
        layers:     Number of SEBottleneck blocks per stage (4-element list).
        groups:     Convolution groups for 3×3 (1=SE-ResNet, 32=SE-ResNeXt).
        base_width: Base width per group (64=SE-ResNet, 4=se-resnext50_32x4d).
        reduction:  SE reduction ratio (default 16).
        img_size, in_chans, drop_rate, out_indices: See BaseVisionBackbone.
    """

    arch_name: str = "se_resnet"
    embed_dim: int = 2048
    feature_dims: list[int] = [256, 512, 1024, 2048]

    def __init__(
        self,
        layers: list[int],
        groups: int = 1,
        base_width: int = 64,
        reduction: int = 16,
        img_size: int = 224,
        in_chans: int = 3,
        drop_rate: float = 0.0,
        out_indices: list[int] | None = None,
    ) -> None:
        super().__init__(
            img_size=img_size, in_chans=in_chans, drop_rate=drop_rate, out_indices=out_indices
        )

        self._groups = groups
        self._base_width = base_width
        self._reduction = reduction
        self._in_planes = 64

        # ── Stem ─────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ── 4 residual stages ─────────────────────────────────────────────────
        self.stage0 = self._make_stage(64, layers[0], stride=1)
        self.stage1 = self._make_stage(128, layers[1], stride=2)
        self.stage2 = self._make_stage(256, layers[2], stride=2)
        self.stage3 = self._make_stage(512, layers[3], stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        self._stages = [self.stage0, self.stage1, self.stage2, self.stage3]
        self._init_weights()

    def _make_stage(
        self,
        planes: int,
        n_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Build one SE-ResNet stage."""
        out_channels = planes * SEBottleneck.expansion
        downsample = None

        if stride != 1 or self._in_planes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self._in_planes, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        blocks = [
            SEBottleneck(
                self._in_planes,
                planes,
                stride,
                downsample,
                groups=self._groups,
                base_width=self._base_width,
                reduction=self._reduction,
            )
        ]
        self._in_planes = out_channels

        for _ in range(1, n_blocks):
            blocks.append(
                SEBottleneck(
                    self._in_planes,
                    planes,
                    groups=self._groups,
                    base_width=self._base_width,
                    reduction=self._reduction,
                )
            )

        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Zero-init the last BN of each SEBottleneck's main branch for stable training
        for m in self.modules():
            if isinstance(m, SEBottleneck):
                nn.init.zeros_(m.bn3.weight)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """Run SE-ResNet forward pass.

        Args:
            x: [B, in_chans, H, W]

        Returns:
            BackboneOutput:
              pooled_output — [B, 2048]
              feature_maps  — strides [4, 8, 16, 32]: [B,256], [B,512], [B,1024], [B,2048]
        """
        x = self.stem(x)

        s0 = self.stage0(x)
        s1 = self.stage1(s0)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)

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
        """Freeze stem and stages 0..stage (inclusive)."""
        if stage < 0 or stage > 3:
            raise ValueError(f"stage must be in [0, 3], got {stage}")
        for p in self.stem.parameters():
            p.requires_grad_(False)
        for i in range(stage + 1):
            for p in self._stages[i].parameters():
                p.requires_grad_(False)
