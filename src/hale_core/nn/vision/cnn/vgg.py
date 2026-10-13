"""
VGGNet Vision Backbones
=======================

Implements the VGG family (Simonyan & Zisserman, ICLR 2015).
VGG demonstrated that depth with small 3×3 convolutions outperforms shallower
networks with larger kernels — a key insight that shaped CNN design for years.

Architecture
------------
All VGG variants share the same pattern: sequential conv-pool stages followed
by a 3-layer FC classifier. The configs differ only in how many conv layers
appear per stage:

  Stage 0: 1–2 × Conv(64) + MaxPool   → H/2
  Stage 1: 1–2 × Conv(128) + MaxPool  → H/4
  Stage 2: 2–4 × Conv(256) + MaxPool  → H/8
  Stage 3: 2–4 × Conv(512) + MaxPool  → H/16
  Stage 4: 2–4 × Conv(512) + MaxPool  → H/32
  Classifier: AdaptiveAvgPool(7,7) → FC(4096) → FC(4096)

_bn variants add BatchNorm2d after every Conv, which wasn't in the original
paper but dramatically improves training stability and final accuracy.

Weight initialisation:
  Convolutions: Kaiming normal.
  Linear layers: truncated normal (std=0.01).

Registered names
----------------
  vgg11, vgg13, vgg16, vgg19           — without BatchNorm
  vgg11_bn, vgg13_bn, vgg16_bn, vgg19_bn — with BatchNorm

Citation
--------
  Simonyan, K., & Zisserman, A. (2015). Very Deep Convolutional Networks for
  Large-Scale Image Recognition. ICLR 2015. arXiv:1409.1556.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.registry import register_vision_backbone

# ── VGG configuration table ───────────────────────────────────────────────────
# Each list entry is either an int (out_channels for a Conv3×3) or 'M' (MaxPool).
# Five MaxPools divide the backbone into 5 stages.
_CFGS: dict[str, list[int | str]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _make_vgg_stages(
    cfg: list[int | str],
    batch_norm: bool,
    in_chans: int = 3,
) -> list[nn.Sequential]:
    """Parse a VGG config into 5 nn.Sequential stage modules.

    Each MaxPool ('M') marks a stage boundary.  We split the config at each
    'M' so that stage.forward() handles one spatial scale.

    Returns list of 5 nn.Sequential modules, one per stage (5 MaxPools total).
    """
    stages: list[nn.Sequential] = []
    layers: list[nn.Module] = []
    in_ch = in_chans

    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            stages.append(nn.Sequential(*layers))
            layers = []
        else:
            assert isinstance(v, int)
            layers.append(nn.Conv2d(in_ch, v, kernel_size=3, padding=1))
            if batch_norm:
                layers.append(nn.BatchNorm2d(v))
            layers.append(nn.ReLU(inplace=True))
            in_ch = v

    # Remaining layers after last MaxPool (shouldn't happen with standard cfgs,
    # but guard against custom configs)
    if layers:
        stages.append(nn.Sequential(*layers))

    return stages


class VGG(BaseVisionBackbone):
    """Generic VGG backbone.

    Args:
        cfg_name:   One of "vgg11", "vgg13", "vgg16", "vgg19".
        batch_norm: Whether to add BatchNorm after each Conv.
        img_size:   Expected input resolution (any size works).
        in_chans:   Input channels.
        drop_rate:  Dropout probability in the FC classifier.
    """

    embed_dim    = 4096
    feature_dims = [64, 128, 256, 512, 512]

    def __init__(
        self,
        cfg_name:   str,
        batch_norm: bool  = False,
        img_size:   int   = 224,
        in_chans:   int   = 3,
        drop_rate:  float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(img_size=img_size, in_chans=in_chans, drop_rate=drop_rate)

        cfg = _CFGS[cfg_name]
        stage_list = _make_vgg_stages(cfg, batch_norm, in_chans=in_chans)

        # Each VGG config has exactly 5 MaxPool separators → 5 stages
        assert len(stage_list) == 5, f"Expected 5 stages, got {len(stage_list)}"
        self.stage0, self.stage1, self.stage2, self.stage3, self.stage4 = stage_list
        self._stage_list = stage_list

        # AdaptiveAvgPool → 7×7 normalises any input resolution for the FC head
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        p = drop_rate if drop_rate > 0.0 else 0.0
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
        )

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
                nn.init.trunc_normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        s0 = self.stage0(x)   # [B, 64,  H/2,  W/2]
        s1 = self.stage1(s0)  # [B, 128, H/4,  W/4]
        s2 = self.stage2(s1)  # [B, 256, H/8,  W/8]
        s3 = self.stage3(s2)  # [B, 512, H/16, W/16]
        s4 = self.stage4(s3)  # [B, 512, H/32, W/32]

        pooled = self.avgpool(s4).flatten(1)    # [B, 512*7*7]
        pooled = self.classifier(pooled)        # [B, 4096]

        all_maps = [s0, s1, s2, s3, s4]
        if self.out_indices is not None:
            all_maps = [all_maps[i] for i in self.out_indices]

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=None,
            feature_maps=all_maps,
            attentions=None,
        )

    def freeze_up_to_stage(self, stage: int) -> None:
        if stage < 0 or stage > 4:
            raise ValueError(f"stage must be in [0, 4], got {stage}")
        for i in range(stage + 1):
            for p in self._stage_list[i].parameters():
                p.requires_grad_(False)


# ── Registered variants ───────────────────────────────────────────────────────

@register_vision_backbone("vgg11")
class VGG11(VGG):
    arch_name = "vgg11"
    def __init__(self, **kwargs): super().__init__("vgg11", batch_norm=False, **kwargs)

@register_vision_backbone("vgg13")
class VGG13(VGG):
    arch_name = "vgg13"
    def __init__(self, **kwargs): super().__init__("vgg13", batch_norm=False, **kwargs)

@register_vision_backbone("vgg16")
class VGG16(VGG):
    arch_name = "vgg16"
    def __init__(self, **kwargs): super().__init__("vgg16", batch_norm=False, **kwargs)

@register_vision_backbone("vgg19")
class VGG19(VGG):
    arch_name = "vgg19"
    def __init__(self, **kwargs): super().__init__("vgg19", batch_norm=False, **kwargs)

@register_vision_backbone("vgg11_bn")
class VGG11BN(VGG):
    arch_name = "vgg11_bn"
    def __init__(self, **kwargs): super().__init__("vgg11", batch_norm=True, **kwargs)

@register_vision_backbone("vgg13_bn")
class VGG13BN(VGG):
    arch_name = "vgg13_bn"
    def __init__(self, **kwargs): super().__init__("vgg13", batch_norm=True, **kwargs)

@register_vision_backbone("vgg16_bn")
class VGG16BN(VGG):
    arch_name = "vgg16_bn"
    def __init__(self, **kwargs): super().__init__("vgg16", batch_norm=True, **kwargs)

@register_vision_backbone("vgg19_bn")
class VGG19BN(VGG):
    arch_name = "vgg19_bn"
    def __init__(self, **kwargs): super().__init__("vgg19", batch_norm=True, **kwargs)
