"""
Inception / GoogLeNet Vision Backbones
=======================================

Implements GoogLeNet / Inception-v1 (Szegedy et al. CVPR 2015) and a clean
modern Inception-v3 (Szegedy et al. CVPR 2016) without auxiliary classifiers.

The key innovation is the Inception module: instead of choosing one filter size,
apply multiple in parallel and concatenate the results.  This captures features
at multiple scales simultaneously within a single layer.

Architecture — GoogLeNet / Inception-v1
-----------------------------------------
  Stem:    Conv(7×7,s2)+BN+ReLU → MaxPool → LRN → Conv(1×1) → Conv(3×3)+BN → MaxPool
  Stage 1: Inception 3a + 3b → MaxPool(3,s2)   → 480 ch, H/16
  Stage 2: Inception 4a–4e   → MaxPool(3,s2)   → 832 ch, H/32
  Stage 3: Inception 5a + 5b → GlobalAvgPool   → [B, 1024]

Architecture — Inception-v3 (clean version, no auxiliary classifiers)
-----------------------------------------------------------------------
  Uses factorised convolutions throughout:
  Type-A: replaces 5×5 with two 3×3
  Type-B: factorises n×n into 1×n + n×1 (asymmetric)
  Type-C: splits 3×3 into 1×3 + 3×1 in parallel (for high-level stages)
  embed_dim=1280.

Registered names
----------------
  inception_v1, inception_v3

Citation
--------
  Szegedy, C., Liu, W., Jia, Y., et al. (2015). Going Deeper with Convolutions.
  CVPR 2015. arXiv:1409.4842.

  Szegedy, C., Vanhoucke, V., Ioffe, S., Shlens, J., & Wojna, Z. (2016).
  Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
  arXiv:1512.00567.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.registry import register_vision_backbone


# ── Shared building blocks ────────────────────────────────────────────────────

class ConvBnRelu(nn.Module):
    """Conv2d + BatchNorm2d + ReLU — the atomic unit in all Inception variants."""
    def __init__(self, in_ch: int, out_ch: int, **conv_kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, bias=False, **conv_kwargs)
        self.bn   = nn.BatchNorm2d(out_ch, eps=0.001)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.bn(self.conv(x)), inplace=True)


# ── Inception-v1 module ───────────────────────────────────────────────────────

class InceptionModuleV1(nn.Module):
    """Inception module as defined in GoogLeNet (Szegedy et al. 2015).

    Four parallel branches computed independently then concatenated on the
    channel dimension, capturing features at multiple receptive field sizes.

    Args:
        in_ch: Input channels.
        c1:    Branch-1 output channels (1×1 conv).
        c2_r:  Branch-2 reduction (1×1 before 3×3).
        c2:    Branch-2 output (3×3 conv).
        c3_r:  Branch-3 reduction (1×1 before 5×5).
        c3:    Branch-3 output (5×5 conv).
        c4:    Branch-4 output (1×1 after MaxPool).
    """
    def __init__(
        self,
        in_ch: int,
        c1: int,
        c2_r: int, c2: int,
        c3_r: int, c3: int,
        c4: int,
    ) -> None:
        super().__init__()
        self.branch1 = ConvBnRelu(in_ch, c1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBnRelu(in_ch, c2_r, kernel_size=1),
            ConvBnRelu(c2_r, c2, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvBnRelu(in_ch, c3_r, kernel_size=1),
            ConvBnRelu(c3_r, c3, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            ConvBnRelu(in_ch, c4, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


# ── GoogLeNet / Inception-v1 ──────────────────────────────────────────────────

class GoogLeNet(BaseVisionBackbone):
    """GoogLeNet (Inception-v1) backbone.

    Returns 4-scale feature maps (stem + one per MaxPool boundary) plus a
    1024-dim global pooled embedding from the final Inception stage.
    """

    arch_name    = "inception_v1"
    embed_dim    = 1024
    feature_dims = [192, 480, 832, 1024]

    def __init__(
        self,
        img_size:  int   = 224,
        in_chans:  int   = 3,
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
            InceptionModuleV1(192, 64,  96, 128, 16,  32, 32),   # 3a → 256
            InceptionModuleV1(256, 128, 128, 192, 32,  96, 64),  # 3b → 480
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Stage 2: Inception 4a–4e (up to 832 ch) → MaxPool → [B, 832, H/32]
        self.stage2 = nn.Sequential(
            InceptionModuleV1(480, 192,  96, 208, 16,  48, 64),  # 4a → 512
            InceptionModuleV1(512, 160, 112, 224, 24,  64, 64),  # 4b → 512
            InceptionModuleV1(512, 128, 128, 256, 24,  64, 64),  # 4c → 512
            InceptionModuleV1(512, 112, 144, 288, 32,  64, 64),  # 4d → 528
            InceptionModuleV1(528, 256, 160, 320, 32, 128, 128), # 4e → 832
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # Stage 3: Inception 5a (832 ch) + 5b (1024 ch) → GlobalAvgPool
        self.stage3 = nn.Sequential(
            InceptionModuleV1(832, 256, 160, 320, 32, 128, 128),  # 5a → 832
            InceptionModuleV1(832, 384, 192, 384, 48, 128, 128),  # 5b → 1024
        )

        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
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
        s0 = self.stem(x)        # [B, 192, H/8, W/8]
        s1 = self.stage1(s0)     # [B, 480, H/16, W/16]
        s2 = self.stage2(s1)     # [B, 832, H/32, W/32]
        s3 = self.stage3(s2)     # [B, 1024, H/32, W/32]

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


# ── Inception-v3 ──────────────────────────────────────────────────────────────

class _ReductionA(nn.Module):
    """Grid reduction A: 288 → 768 channels with stride 2."""
    def __init__(self, in_ch: int = 288) -> None:
        super().__init__()
        # Branch 1: 3×3 stride-2 conv
        self.br1 = ConvBnRelu(in_ch, 384, kernel_size=3, stride=2)
        # Branch 2: 1×1 → 3×3 → 3×3 stride-2
        self.br2 = nn.Sequential(
            ConvBnRelu(in_ch, 64, kernel_size=1),
            ConvBnRelu(64, 96, kernel_size=3, padding=1),
            ConvBnRelu(96, 96, kernel_size=3, stride=2),
        )
        # Branch 3: MaxPool stride-2 (keeps in_ch channels)
        self.br3 = nn.MaxPool2d(kernel_size=3, stride=2)
        # Total: 384 + 96 + 288 = 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.br1(x), self.br2(x), self.br3(x)], dim=1)


class _ReductionB(nn.Module):
    """Grid reduction B: 768 → 1280 channels with stride 2."""
    def __init__(self, in_ch: int = 768) -> None:
        super().__init__()
        self.br1 = nn.Sequential(
            ConvBnRelu(in_ch, 192, kernel_size=1),
            ConvBnRelu(192, 320, kernel_size=3, stride=2),
        )
        self.br2 = nn.Sequential(
            ConvBnRelu(in_ch, 192, kernel_size=1),
            ConvBnRelu(192, 192, kernel_size=(1, 7), padding=(0, 3)),
            ConvBnRelu(192, 192, kernel_size=(7, 1), padding=(3, 0)),
            ConvBnRelu(192, 192, kernel_size=3, stride=2),
        )
        self.br3 = nn.MaxPool2d(kernel_size=3, stride=2)
        # Total: 320 + 192 + 768 = 1280

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.br1(x), self.br2(x), self.br3(x)], dim=1)


class _InceptionTypeA(nn.Module):
    """Inception-v3 Type-A: replaces 5×5 with two 3×3 convs."""
    def __init__(self, in_ch: int, pool_features: int) -> None:
        super().__init__()
        self.br1 = ConvBnRelu(in_ch, 64, kernel_size=1)
        self.br2 = nn.Sequential(
            ConvBnRelu(in_ch, 48, kernel_size=1),
            ConvBnRelu(48, 64, kernel_size=5, padding=2),
        )
        self.br3 = nn.Sequential(
            ConvBnRelu(in_ch, 64, kernel_size=1),
            ConvBnRelu(64, 96, kernel_size=3, padding=1),
            ConvBnRelu(96, 96, kernel_size=3, padding=1),
        )
        self.br4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBnRelu(in_ch, pool_features, kernel_size=1),
        )
        # Total: 64+64+96+pool_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.br1(x), self.br2(x), self.br3(x), self.br4(x)], dim=1)


class _InceptionTypeB(nn.Module):
    """Inception-v3 Type-B: factorised n×n → 1×n + n×1 (asymmetric)."""
    def __init__(self, in_ch: int, channels_7x7: int) -> None:
        super().__init__()
        c7 = channels_7x7
        self.br1 = ConvBnRelu(in_ch, 192, kernel_size=1)
        self.br2 = nn.Sequential(
            ConvBnRelu(in_ch, c7, kernel_size=1),
            ConvBnRelu(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
            ConvBnRelu(c7, 192, kernel_size=(7, 1), padding=(3, 0)),
        )
        self.br3 = nn.Sequential(
            ConvBnRelu(in_ch, c7, kernel_size=1),
            ConvBnRelu(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            ConvBnRelu(c7, c7, kernel_size=(1, 7), padding=(0, 3)),
            ConvBnRelu(c7, c7, kernel_size=(7, 1), padding=(3, 0)),
            ConvBnRelu(c7, 192, kernel_size=(1, 7), padding=(0, 3)),
        )
        self.br4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBnRelu(in_ch, 192, kernel_size=1),
        )
        # Total: 192 * 4 = 768

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.br1(x), self.br2(x), self.br3(x), self.br4(x)], dim=1)


class _InceptionTypeC(nn.Module):
    """Inception-v3 Type-C: splits 3×3 into 1×3 + 3×1 in parallel (wide).
    Used in deepest stage for maximum filter bank diversity."""
    def __init__(self, in_ch: int) -> None:
        super().__init__()
        self.br1 = ConvBnRelu(in_ch, 320, kernel_size=1)
        self.br2_r   = ConvBnRelu(in_ch, 384, kernel_size=1)
        self.br2_1x3 = ConvBnRelu(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.br2_3x1 = ConvBnRelu(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.br3_r   = ConvBnRelu(in_ch, 448, kernel_size=1)
        self.br3_3x3 = ConvBnRelu(448, 384, kernel_size=3, padding=1)
        self.br3_1x3 = ConvBnRelu(384, 384, kernel_size=(1, 3), padding=(0, 1))
        self.br3_3x1 = ConvBnRelu(384, 384, kernel_size=(3, 1), padding=(1, 0))
        self.br4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            ConvBnRelu(in_ch, 192, kernel_size=1),
        )
        # Total: 320 + 384+384 + 384+384 + 192 = 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1   = self.br1(x)
        b2_r = self.br2_r(x)
        b2   = torch.cat([self.br2_1x3(b2_r), self.br2_3x1(b2_r)], dim=1)  # 768
        b3_r = self.br3_3x3(self.br3_r(x))
        b3   = torch.cat([self.br3_1x3(b3_r), self.br3_3x1(b3_r)], dim=1)  # 768
        b4   = self.br4(x)
        return torch.cat([b1, b2, b3, b4], dim=1)  # 320+768+768+192 = 2048


class InceptionV3(BaseVisionBackbone):
    """Inception-v3 backbone (no auxiliary classifiers).

    Factorised convolutions throughout; embed_dim=2048 from mixed_7c.
    """

    arch_name    = "inception_v3"
    embed_dim    = 2048
    feature_dims = [192, 768, 1280, 2048]

    def __init__(
        self,
        img_size:  int   = 299,
        in_chans:  int   = 3,
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
            _InceptionTypeA(192, pool_features=32),   # → 256
            _InceptionTypeA(256, pool_features=64),   # → 288
            _InceptionTypeA(288, pool_features=64),   # → 288
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
            _InceptionTypeC(in_ch=2048), # → 2048
        )

        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
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
        s0 = self.stem(x)            # [B, 192, ~H/32, ~W/32]
        s1 = self.reduction_a(self.stage1(s0))   # [B, 768, ...]
        s2 = self.reduction_b(self.stage2(s1))   # [B, 1280, ...]
        s3 = self.stage3(s2)         # [B, 2048, ...]

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


# ── Registered variants ───────────────────────────────────────────────────────

@register_vision_backbone("inception_v1")
class InceptionV1Backbone(GoogLeNet):
    """GoogLeNet / Inception-v1: 22 layers, embed_dim=1024."""
    arch_name = "inception_v1"
    def __init__(self, **kwargs): super().__init__(**kwargs)


@register_vision_backbone("inception_v3")
class InceptionV3Backbone(InceptionV3):
    """Inception-v3: factorised convolutions, embed_dim=2048."""
    arch_name = "inception_v3"
    def __init__(self, **kwargs): super().__init__(**kwargs)
