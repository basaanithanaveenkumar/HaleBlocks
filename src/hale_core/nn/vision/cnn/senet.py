"""
SENet / SE-ResNet Vision Backbones
=====================================

Implements SE-ResNet variants (Hu et al. 2018, "Squeeze-and-Excitation Networks",
CVPR 2018, ILSVRC 2017 classification winner).

Reference: https://arxiv.org/abs/1709.01507

Architecture
------------
Squeeze-and-Excitation (SE) adds a channel attention mechanism to residual blocks.
The SE block recalibrates channel-wise feature responses by:

  1. Squeeze:   GlobalAvgPool [B, C, H, W] → [B, C, 1, 1]
                Collapses spatial information into a channel descriptor.
  2. Excitation: Two FC layers with a bottleneck:
                 FC(C→C//r) → ReLU → FC(C//r→C) → Sigmoid → [B, C, 1, 1]
                 Learns non-linear channel interdependencies.
  3. Scale:     x × scale — multiplies the input feature map channel-wise.

The SE block is inserted after the last conv in a Bottleneck block (before the
residual addition), giving the network "self-recalibration" ability at each stage.

Parameter overhead: 2C²/r per block.  For C=256, r=16: ~8192 extra params per block.

Stage structure (same as ResNet-50/101/152):
  Stem    : 7×7 Conv + BN + ReLU + MaxPool (s2+s2) → 1/4 resolution
  Stage 0 : SEBottleneck × [3, 3, 3], 64 planes,  stride=1  → [B, 256,  H/4, W/4]
  Stage 1 : SEBottleneck × [4, 4, 4], 128 planes, stride=2  → [B, 512,  H/8, W/8]
  Stage 2 : SEBottleneck × [6,23,36], 256 planes, stride=2  → [B, 1024, H/16,W/16]
  Stage 3 : SEBottleneck × [3, 3, 3], 512 planes, stride=2  → [B, 2048, H/32,W/32]

SE-ResNeXt variant: replaces the grouped 3×3 conv in the bottleneck
  (cardinality C=32, base_width=4 for seresnext50_32x4d).

Registered names
----------------
  se_resnet50       — SEBottleneck × [3, 4, 6, 3],  embed_dim=2048
  se_resnet101      — SEBottleneck × [3, 4, 23, 3], embed_dim=2048
  se_resnet152      — SEBottleneck × [3, 8, 36, 3], embed_dim=2048
  seresnext50_32x4d — SE-ResNeXt (cardinality=32, base_width=4), embed_dim=2048

Citation
--------
  Hu, J., Shen, L., & Sun, G. (2018). Squeeze-and-Excitation Networks. CVPR.
  arXiv:1709.01507.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.registry import register_vision_backbone


# ── Squeeze-Excitation block ──────────────────────────────────────────────────

class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention block.

    Reduces channels by reduction ratio r, then re-expands, producing a per-channel
    scaling vector in [0, 1] via Sigmoid. This vector multiplies the input features.

    Args:
        channels:  Input (and output) channel count.
        reduction: Squeeze ratio r.  Default 16 (original paper setting).
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        squeezed = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        # Two linear layers implemented as 1×1 convolutions (avoids reshape)
        self.fc1  = nn.Conv2d(channels, squeezed, kernel_size=1, bias=True)
        self.fc2  = nn.Conv2d(squeezed, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze: global spatial average
        s = self.pool(x)                        # [B, C, 1, 1]
        # Excitation: FC → ReLU → FC → Sigmoid
        s = F.relu(self.fc1(s), inplace=True)   # [B, C//r, 1, 1]
        s = torch.sigmoid(self.fc2(s))          # [B, C,    1, 1]
        # Scale: channel-wise multiplication
        return x * s


# ── SE Bottleneck block ───────────────────────────────────────────────────────

class SEBottleneck(nn.Module):
    """ResNet Bottleneck with SE channel attention.

    Architecture:
        input → Conv1×1 → BN → ReLU
              → Conv3×3 → BN → ReLU           (groups for ResNeXt variant)
              → Conv1×1 → BN
              → SE(reduction=16)               ← channel attention here
              → + residual → ReLU

    Args:
        in_planes:  Input channels.
        planes:     Bottleneck width (output = planes × 4).
        stride:     Applied to the 3×3 conv for spatial downsampling.
        downsample: 1×1 projection shortcut for shape matching.
        groups:     Convolution groups for the 3×3 (1=ResNet, 32=ResNeXt).
        base_width: Base channel width per group for ResNeXt scaling.
        reduction:  SE reduction ratio.
    """

    expansion: int = 4

    def __init__(
        self,
        in_planes:  int,
        planes:     int,
        stride:     int               = 1,
        downsample: Optional[nn.Module] = None,
        groups:     int               = 1,
        base_width: int               = 64,
        reduction:  int               = 16,
    ) -> None:
        super().__init__()
        # ResNeXt-style width scaling: actual mid-channels depend on groups + base_width
        width = int(planes * (base_width / 64.0)) * groups
        out_channels = planes * self.expansion

        # 1×1 reduce
        self.conv1 = nn.Conv2d(in_planes, width, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(width)

        # 3×3 spatial (with optional grouping for ResNeXt)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=groups, bias=False)
        self.bn2   = nn.BatchNorm2d(width)

        # 1×1 expand
        self.conv3 = nn.Conv2d(width, out_channels, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels)

        # SE block applied to the expanded feature map
        self.se = SEBlock(out_channels, reduction=reduction)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))  # no activation before SE+add

        # SE recalibration before residual addition
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = F.relu(out + identity, inplace=True)
        return out


# ── Core SE-ResNet model ──────────────────────────────────────────────────────

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

    arch_name:    str       = "se_resnet"
    embed_dim:    int       = 2048
    feature_dims: list[int] = [256, 512, 1024, 2048]

    def __init__(
        self,
        layers:      list[int],
        groups:      int  = 1,
        base_width:  int  = 64,
        reduction:   int  = 16,
        img_size:    int  = 224,
        in_chans:    int  = 3,
        drop_rate:   float = 0.0,
        out_indices: list[int] | None = None,
    ) -> None:
        super().__init__(img_size=img_size, in_chans=in_chans,
                         drop_rate=drop_rate, out_indices=out_indices)

        self._groups     = groups
        self._base_width = base_width
        self._reduction  = reduction
        self._in_planes  = 64

        # ── Stem ─────────────────────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ── 4 residual stages ─────────────────────────────────────────────────
        self.stage0 = self._make_stage(64,  layers[0], stride=1)
        self.stage1 = self._make_stage(128, layers[1], stride=2)
        self.stage2 = self._make_stage(256, layers[2], stride=2)
        self.stage3 = self._make_stage(512, layers[3], stride=2)

        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        self._stages = [self.stage0, self.stage1, self.stage2, self.stage3]
        self._init_weights()

    def _make_stage(
        self,
        planes:   int,
        n_blocks: int,
        stride:   int,
    ) -> nn.Sequential:
        """Build one SE-ResNet stage."""
        out_channels = planes * SEBottleneck.expansion
        downsample   = None

        if stride != 1 or self._in_planes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self._in_planes, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        blocks = [SEBottleneck(
            self._in_planes, planes, stride, downsample,
            groups=self._groups, base_width=self._base_width, reduction=self._reduction,
        )]
        self._in_planes = out_channels

        for _ in range(1, n_blocks):
            blocks.append(SEBottleneck(
                self._in_planes, planes,
                groups=self._groups, base_width=self._base_width, reduction=self._reduction,
            ))

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


# ── Registered variants ───────────────────────────────────────────────────────

@register_vision_backbone("se_resnet50")
class SEResNet50(SEResNet):
    """SE-ResNet-50: SEBottleneck × [3,4,6,3], ~28M params, embed_dim=2048."""
    arch_name    = "se_resnet50"
    embed_dim    = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__(layers=[3, 4, 6, 3], **kwargs)


@register_vision_backbone("se_resnet101")
class SEResNet101(SEResNet):
    """SE-ResNet-101: SEBottleneck × [3,4,23,3], ~49M params, embed_dim=2048."""
    arch_name    = "se_resnet101"
    embed_dim    = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__(layers=[3, 4, 23, 3], **kwargs)


@register_vision_backbone("se_resnet152")
class SEResNet152(SEResNet):
    """SE-ResNet-152: SEBottleneck × [3,8,36,3], ~66M params, embed_dim=2048."""
    arch_name    = "se_resnet152"
    embed_dim    = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__(layers=[3, 8, 36, 3], **kwargs)


@register_vision_backbone("seresnext50_32x4d")
class SEResNeXt50_32x4d(SEResNet):
    """SE-ResNeXt-50 (32×4d): cardinality=32, base_width=4, ~27M params."""
    arch_name    = "seresnext50_32x4d"
    embed_dim    = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__(layers=[3, 4, 6, 3], groups=32, base_width=4, **kwargs)
