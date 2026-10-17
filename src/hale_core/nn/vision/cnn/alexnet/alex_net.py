"""``AlexNet``: AlexNet backbone adapted for dense feature extraction."""

from __future__ import annotations

import torch
import torch.nn as nn

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone


class AlexNet(BaseVisionBackbone):
    """AlexNet backbone adapted for dense feature extraction.

    Returns 5 stage feature maps (from each conv block) plus a 4096-dim
    pooled embedding from the FC layers.

    Args:
        img_size:  Expected input spatial resolution (any size works due to AdaptiveAvgPool).
        in_chans:  Input channels (default 3).
        drop_rate: Dropout probability in the FC classifier.
    """

    arch_name = "alexnet"
    embed_dim = 4096
    feature_dims = [64, 192, 384, 256, 256]

    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 3,
        drop_rate: float = 0.5,
        **kwargs,
    ) -> None:
        super().__init__(img_size=img_size, in_chans=in_chans, drop_rate=drop_rate)

        # Stage 0: large 11×11 receptive field at stride 4 to aggressively downsample
        # Original AlexNet used stride=4 on 227×227 input; AdaptiveAvgPool handles any size
        self.stage0 = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Stage 1: 5×5 conv to capture mid-level texture
        self.stage1 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Stages 2–4: stack of 3×3 convs for fine-grained features
        self.stage2 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Adaptive pool normalises varying input sizes to 6×6 before the FC classifier
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        p = drop_rate if drop_rate > 0.0 else 0.0
        self.classifier = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )

        self._stages = [self.stage0, self.stage1, self.stage2, self.stage3, self.stage4]
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                # Original paper: normal(0, 0.01); trunc_normal avoids rare large inits
                nn.init.trunc_normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """Forward pass of :class:`AlexNet`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``BackboneOutput``.
        """
        s0 = self.stage0(x)  # [B, 64,  H/8,  W/8]
        s1 = self.stage1(s0)  # [B, 192, H/16, W/16]
        s2 = self.stage2(s1)  # [B, 384, H/16, W/16]
        s3 = self.stage3(s2)  # [B, 256, H/16, W/16]
        s4 = self.stage4(s3)  # [B, 256, H/32, W/32]

        pooled = self.avgpool(s4)  # [B, 256, 6, 6]
        pooled = pooled.flatten(1)  # [B, 9216]
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
        """Freeze the parameters of stages ``0 … stage`` (stem first) for staged fine-tuning.

        Raises:
            ValueError: if ``stage`` is outside this backbone's stage range.
        """
        if stage < 0 or stage > 4:
            raise ValueError(f"stage must be in [0, 4], got {stage}")
        for i in range(stage + 1):
            for p in self._stages[i].parameters():
                p.requires_grad_(False)
