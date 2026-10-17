"""``DenseNet``: Generic DenseNet backbone."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.cnn.densenet._helpers import _Transition
from hale_core.nn.vision.cnn.densenet.dense_block import DenseBlock


class DenseNet(BaseVisionBackbone):
    """Generic DenseNet backbone.

    Returns 4 multi-scale feature maps (one per DenseBlock output, before Transition
    pooling) plus a global pooled embedding after the final block.

    Args:
        block_config:  Number of layers per DenseBlock, e.g. (6, 12, 24, 16).
        growth_rate:   k — channels added per dense layer (default 32).
        init_features: Channels after initial conv (= 2*k = 64).
        img_size:      Input resolution.
        in_chans:      Input channels.
        drop_rate:     Dropout after global pooling.
    """

    def __init__(
        self,
        block_config: tuple[int, ...],
        growth_rate: int = 32,
        init_features: int = 64,
        img_size: int = 224,
        in_chans: int = 3,
        drop_rate: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(img_size=img_size, in_chans=in_chans, drop_rate=drop_rate)

        k = growth_rate

        # Initial convolution: 3 → 2k channels, stride-2 + MaxPool stride-2
        self.initial = nn.Sequential(
            nn.Conv2d(in_chans, init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        num_features = init_features
        self.dense_blocks: nn.ModuleList = nn.ModuleList()
        self.transitions: nn.ModuleList = nn.ModuleList()
        feature_dims: list[int] = []

        for i, num_layers in enumerate(block_config):
            block = DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=k,
            )
            self.dense_blocks.append(block)
            num_features = block.out_channels
            feature_dims.append(num_features)

            if i != len(block_config) - 1:
                # Transition compresses channels by 0.5 and halves spatial resolution
                out_ch = num_features // 2
                trans = _Transition(num_features, out_ch)
                self.transitions.append(trans)
                num_features = out_ch
            else:
                # No transition after the last block
                self.transitions.append(None)  # placeholder for indexing consistency

        # Final BN before pooling
        self.final_norm = nn.BatchNorm2d(num_features)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        self.embed_dim = num_features
        self.feature_dims = feature_dims  # channels after each DenseBlock (before transition pool)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """Forward pass of :class:`DenseNet`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``BackboneOutput``.
        """
        x = self.initial(x)  # [B, init_features, H/4, W/4]

        feature_maps = []
        for i, (block, trans) in enumerate(zip(self.dense_blocks, self.transitions)):
            x = block(x)
            # Capture feature map BEFORE transition's AvgPool for FPN compatibility
            feature_maps.append(x)

            if trans is not None:
                x = trans(x)  # halves spatial + channel dims

        # Final normalisation and global pooling
        x = F.relu(self.final_norm(x), inplace=True)
        pooled = self.dropout(self.pool(x).flatten(1))  # [B, embed_dim]

        if self.out_indices is not None:
            feature_maps = [feature_maps[i] for i in self.out_indices]

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=None,
            feature_maps=feature_maps,
            attentions=None,
        )

    def freeze_up_to_stage(self, stage: int) -> None:
        if stage < 0 or stage >= len(self.dense_blocks):
            raise ValueError(f"stage must be in [0, {len(self.dense_blocks) - 1}], got {stage}")
        # Freeze initial stem
        for p in self.initial.parameters():
            p.requires_grad_(False)
        for i in range(stage + 1):
            for p in self.dense_blocks[i].parameters():
                p.requires_grad_(False)
            if self.transitions[i] is not None:
                for p in self.transitions[i].parameters():
                    p.requires_grad_(False)
