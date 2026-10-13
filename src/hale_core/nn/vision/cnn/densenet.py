"""
DenseNet Vision Backbones
=========================

Implements the DenseNet family (Huang et al. CVPR 2017, "Densely Connected
Convolutional Networks").

The key innovation: each layer is connected to ALL subsequent layers in its
dense block.  Instead of a single skip connection (ResNet), features are
concatenated — every layer has direct access to all prior feature maps,
maximising feature reuse and enabling very narrow layers (growth rate k=32)
to be competitive with much wider non-dense networks.

Architecture
------------
  Initial conv: Conv(3→2k, 7×7, s2, pad3) + BN + ReLU + MaxPool(3,s2)

  DenseBlock: each of L layers produces k new channels via:
    BN → ReLU → Conv1×1(in→4k) → BN → ReLU → Conv3×3(4k→k)
    Input to layer ℓ = concat of all prior layer outputs + initial input

  Transition: compresses and downsamples between blocks:
    BN → ReLU → Conv1×1(C→C//2) → AvgPool(2,s2)

  Four DenseBlocks separated by three Transitions, then:
    BN → ReLU → AdaptiveAvgPool → Linear → embed_dim

Feature map extraction: we return one feature map per DenseBlock output
(before the Transition's AvgPool), giving 4 spatial scales.

Registered names
----------------
  densenet121 — layers=(6,12,24,16), embed_dim=1024
  densenet169 — layers=(6,12,32,32), embed_dim=1664
  densenet201 — layers=(6,12,48,32), embed_dim=1920
  densenet264 — layers=(6,12,64,48), embed_dim=2688

Citation
--------
  Huang, G., Liu, Z., van der Maaten, L., & Weinberger, K. Q. (2017).
  Densely Connected Convolutional Networks. CVPR 2017. arXiv:1608.06993.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.registry import register_vision_backbone


# ── DenseBlock components ─────────────────────────────────────────────────────

class _DenseLayer(nn.Module):
    """Single layer in a DenseBlock.

    Implements the BN-ReLU-Conv1×1-BN-ReLU-Conv3×3 "bottleneck" design.
    Input is ALL prior layer outputs concatenated; output is k new channels.

    The 1×1 conv compresses the accumulated channels to 4k before the 3×3 conv,
    keeping computation tractable as the concatenation dimension grows.

    Args:
        num_input_features: Number of input channels (all prior features).
        growth_rate:        k — number of new channels this layer produces.
        memory_efficient:   Use gradient checkpointing (not implemented here).
    """
    def __init__(self, num_input_features: int, growth_rate: int) -> None:
        super().__init__()
        inter_ch = 4 * growth_rate  # 1×1 conv target = 4k

        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.conv1 = nn.Conv2d(num_input_features, inter_ch, kernel_size=1, bias=False)
        self.norm2 = nn.BatchNorm2d(inter_ch)
        self.conv2 = nn.Conv2d(inter_ch, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Typical implementation receives a list of all previous outputs which
        # are concatenated inside forward — this matches DenseNet paper appendix.
        out = self.conv1(F.relu(self.norm1(x), inplace=True))
        out = self.conv2(F.relu(self.norm2(out), inplace=True))
        return out


class DenseBlock(nn.Module):
    """Dense block: L layers with all-pairs skip connections.

    Each layer's input is the concatenation of ALL previous layer outputs
    plus the block's initial input.  This means layer ℓ has
    in_ch + ℓ * growth_rate input channels.

    Args:
        num_layers:          L — number of dense layers.
        num_input_features:  Channels at block entry.
        growth_rate:         k — channels added per layer.
    """
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(_DenseLayer(
                num_input_features=num_input_features + i * growth_rate,
                growth_rate=growth_rate,
            ))
        # Channels after this block
        self.out_channels = num_input_features + num_layers * growth_rate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accumulate all feature maps in a list and concat for each new layer
        features = [x]
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)
        return torch.cat(features, dim=1)  # [B, num_input + L*k, H, W]


class _Transition(nn.Module):
    """Transition layer between DenseBlocks.

    Halves both channel count (via 1×1 conv) and spatial resolution (AvgPool).
    The compression factor of 0.5 is fixed per the original paper (θ=0.5).

    Args:
        num_input_features:  Input channels (= DenseBlock output channels).
        num_output_features: Output channels (= num_input_features // 2).
    """
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.conv(F.relu(self.norm(x), inplace=True)))


# ── Core DenseNet model ───────────────────────────────────────────────────────

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
        block_config:  tuple[int, ...],
        growth_rate:   int   = 32,
        init_features: int   = 64,
        img_size:      int   = 224,
        in_chans:      int   = 3,
        drop_rate:     float = 0.0,
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
        self.transitions:  nn.ModuleList = nn.ModuleList()
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
        self.pool       = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout    = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        self.embed_dim    = num_features
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
            raise ValueError(f"stage must be in [0, {len(self.dense_blocks)-1}], got {stage}")
        # Freeze initial stem
        for p in self.initial.parameters():
            p.requires_grad_(False)
        for i in range(stage + 1):
            for p in self.dense_blocks[i].parameters():
                p.requires_grad_(False)
            if self.transitions[i] is not None:
                for p in self.transitions[i].parameters():
                    p.requires_grad_(False)


# ── Registered variants ───────────────────────────────────────────────────────

@register_vision_backbone("densenet121")
class DenseNet121(DenseNet):
    """DenseNet-121: (6,12,24,16) layers, k=32, embed_dim=1024."""
    arch_name = "densenet121"
    embed_dim = 1024
    feature_dims = [256, 512, 1024, 1024]

    def __init__(self, **kwargs):
        super().__init__(block_config=(6, 12, 24, 16), growth_rate=32, init_features=64, **kwargs)


@register_vision_backbone("densenet169")
class DenseNet169(DenseNet):
    """DenseNet-169: (6,12,32,32) layers, k=32, embed_dim=1664."""
    arch_name = "densenet169"
    embed_dim = 1664
    feature_dims = [256, 512, 1280, 1664]

    def __init__(self, **kwargs):
        super().__init__(block_config=(6, 12, 32, 32), growth_rate=32, init_features=64, **kwargs)


@register_vision_backbone("densenet201")
class DenseNet201(DenseNet):
    """DenseNet-201: (6,12,48,32) layers, k=32, embed_dim=1920."""
    arch_name = "densenet201"
    embed_dim = 1920
    feature_dims = [256, 512, 1792, 1920]

    def __init__(self, **kwargs):
        super().__init__(block_config=(6, 12, 48, 32), growth_rate=32, init_features=64, **kwargs)


@register_vision_backbone("densenet264")
class DenseNet264(DenseNet):
    """DenseNet-264: (6,12,64,48) layers, k=32, embed_dim=2688."""
    arch_name = "densenet264"
    embed_dim = 2688
    feature_dims = [256, 512, 2112, 2688]

    def __init__(self, **kwargs):
        super().__init__(block_config=(6, 12, 64, 48), growth_rate=32, init_features=64, **kwargs)
