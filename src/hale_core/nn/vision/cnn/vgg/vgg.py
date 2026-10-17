"""``VGG``: Generic VGG backbone."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.cnn.vgg._helpers import _CFGS, _make_vgg_stages


class VGG(BaseVisionBackbone):
    """Generic VGG backbone.

    Args:
        cfg_name:   One of "vgg11", "vgg13", "vgg16", "vgg19".
        batch_norm: Whether to add BatchNorm after each Conv.
        img_size:   Expected input resolution (any size works).
        in_chans:   Input channels.
        drop_rate:  Dropout probability in the FC classifier.
    """

    embed_dim = 4096
    feature_dims = [64, 128, 256, 512, 512]

    def __init__(
        self,
        cfg_name: str,
        batch_norm: bool = False,
        img_size: int = 224,
        in_chans: int = 3,
        drop_rate: float = 0.5,
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
        """Forward pass of :class:`VGG`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``BackboneOutput``.
        """
        s0 = self.stage0(x)  # [B, 64,  H/2,  W/2]
        s1 = self.stage1(s0)  # [B, 128, H/4,  W/4]
        s2 = self.stage2(s1)  # [B, 256, H/8,  W/8]
        s3 = self.stage3(s2)  # [B, 512, H/16, W/16]
        s4 = self.stage4(s3)  # [B, 512, H/32, W/32]

        pooled = self.avgpool(s4).flatten(1)  # [B, 512*7*7]
        pooled = self.classifier(pooled)  # [B, 4096]

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
