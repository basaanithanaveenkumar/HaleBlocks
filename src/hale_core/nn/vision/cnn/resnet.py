"""
ResNet Vision Backbones
=======================

Implements the full ResNet family (He et al. 2016, "Deep Residual Learning for Image
Recognition", https://arxiv.org/abs/1512.03385).

Architecture
------------
ResNet is built from stacked residual blocks of two designs:

  BasicBlock (used in ResNet-18 and ResNet-34):
    3×3 conv → BN → ReLU → 3×3 conv → BN → (+residual) → ReLU
    Expansion factor = 1 (output channels == input channels).

  Bottleneck (used in ResNet-50/101/152/200):
    1×1 conv (reduce) → BN → ReLU
    3×3 conv          → BN → ReLU
    1×1 conv (expand) → BN → (+residual) → ReLU
    Expansion factor = 4 (output channels = planes × 4).

Stage structure:
  Stem    : 7×7 conv, stride 2 → BN → ReLU → MaxPool stride 2  → 1/4 resolution
  Stage 0 : 64  channels, stride 1                               → 1/4 resolution
  Stage 1 : 128 channels, stride 2                               → 1/8 resolution
  Stage 2 : 256 channels, stride 2                               → 1/16 resolution
  Stage 3 : 512 channels, stride 2                               → 1/32 resolution

Weight initialisation:
  Convolutions: Kaiming normal (He init) for correct variance under ReLU.
  BatchNorm:    weight=1, bias=0 (standard identity init).
  Zero-init of the last BN in each residual branch (Goyal et al. "Accurate, Large
  Minibatch SGD"), which improves convergence stability at large batch sizes.

Registered names
----------------
  resnet18   — BasicBlock × [2, 2, 2, 2],  embed_dim=512
  resnet34   — BasicBlock × [3, 4, 6, 3],  embed_dim=512
  resnet50   — Bottleneck × [3, 4, 6, 3],  embed_dim=2048
  resnet101  — Bottleneck × [3, 4, 23, 3], embed_dim=2048
  resnet152  — Bottleneck × [3, 8, 36, 3], embed_dim=2048
  resnet200  — Bottleneck × [3, 24, 36, 3],embed_dim=2048

Citation
--------
  He et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
  Goyal et al. (2017). Accurate, Large Minibatch SGD. arXiv:1706.02677.
"""

from __future__ import annotations

from typing import Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.registry import register_vision_backbone


# ── Building blocks ───────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    """Residual block for ResNet-18/34.

    Uses two 3×3 convolutions without channel bottlenecking (expansion=1).
    This design is cheaper per block but stacks more layers to reach target capacity.

    Args:
        in_planes:  Number of input channels.
        planes:     Number of output channels (== in_planes × expansion = in_planes).
        stride:     Stride for the first conv; when >1, spatial downsampling occurs.
        downsample: Optional projection shortcut when input/output shapes differ.
    """

    expansion: int = 1  # output channels = planes * expansion

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        # First 3×3 conv; stride here performs spatial downsampling if stride>1
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        # Second 3×3 conv; always stride=1 to preserve resolution after first conv
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        # Shortcut: may be None (identity) or a 1×1 projection to match dims
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))

        # Add shortcut BEFORE final activation — core residual learning formula
        if self.downsample is not None:
            identity = self.downsample(x)

        out = F.relu(out + identity, inplace=True)
        return out


class Bottleneck(nn.Module):
    """Residual bottleneck block for ResNet-50/101/152/200.

    Uses a 1×1 → 3×3 → 1×1 pattern (expansion=4), significantly reducing FLOPs
    compared to two stacked 3×3 convolutions when channel counts are large.

    Args:
        in_planes:  Number of input channels.
        planes:     "bottleneck width"; output channels = planes × expansion = planes × 4.
        stride:     Stride for the 3×3 conv; when >1, spatial downsampling occurs.
        downsample: Optional projection shortcut when input/output shapes differ.
    """

    expansion: int = 4  # output channels = planes * 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        out_channels = planes * self.expansion

        # 1×1 conv: reduce channels from in_planes to planes (the "bottleneck")
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)

        # 3×3 conv: main spatial processing; stride here for downsampling
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)

        # 1×1 conv: expand channels back to planes × 4
        self.conv3 = nn.Conv2d(planes, out_channels, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))  # no activation before add

        if self.downsample is not None:
            identity = self.downsample(x)

        out = F.relu(out + identity, inplace=True)
        return out


# ── Core ResNet model ─────────────────────────────────────────────────────────

class ResNet(BaseVisionBackbone):
    """Generic ResNet backbone.

    Implements the full multi-stage ResNet as a dense feature extractor, returning
    4-scale feature maps for use with FPN/BEV heads and a global pooled embedding
    for VLM/action encoders.

    Args:
        block:      Block class — BasicBlock or Bottleneck.
        layers:     Number of blocks in each of the 4 stages.
        img_size:   Expected input spatial resolution (assumed square).
        in_chans:   Input image channels (default 3 for RGB).
        drop_rate:  Dropout probability applied after global pooling.
        out_indices: Which stages to include in feature_maps (default: all 4).
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: list[int],
        img_size:    int        = 224,
        in_chans:    int        = 3,
        drop_rate:   float      = 0.0,
        out_indices: list[int] | None = None,
    ) -> None:
        super().__init__(img_size=img_size, in_chans=in_chans, drop_rate=drop_rate, out_indices=out_indices)

        self._block = block
        self._layers = layers

        # embed_dim is the channel count of stage 3 output
        self.embed_dim  = 512 * block.expansion
        # feature_dims matches channels at each of the 4 stages
        self.feature_dims = [
            64  * block.expansion if block.expansion > 1 else 64,
            128 * block.expansion if block.expansion > 1 else 128,
            256 * block.expansion if block.expansion > 1 else 256,
            512 * block.expansion if block.expansion > 1 else 512,
        ]
        # For BasicBlock (expansion=1), channels are [64, 128, 256, 512]
        # For Bottleneck (expansion=4), channels are [256, 512, 1024, 2048]
        self.feature_dims = [
            64,
            128,
            256,
            512,
        ] if block.expansion == 1 else [256, 512, 1024, 2048]

        self._in_planes = 64  # current channel count, updated as stages are built

        # ── Stem: 7×7 conv + BN + ReLU + MaxPool ─────────────────────────────
        # Downsamples input from H × W to H/4 × W/4 before stage 0.
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        # ── 4 residual stages ─────────────────────────────────────────────────
        # Stage 0: stride=1 (no downsampling, maintains H/4)
        self.stage0 = self._make_stage(block,  64, layers[0], stride=1)
        # Stages 1–3: stride=2 each, progressively halving resolution
        self.stage1 = self._make_stage(block, 128, layers[1], stride=2)
        self.stage2 = self._make_stage(block, 256, layers[2], stride=2)
        self.stage3 = self._make_stage(block, 512, layers[3], stride=2)

        # ── Global average pool + optional dropout ────────────────────────────
        self.pool    = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else nn.Identity()

        # Ordered list of stage modules for freeze_up_to_stage convenience
        self._stages = [self.stage0, self.stage1, self.stage2, self.stage3]

        self._init_weights()

    def _make_stage(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        n_blocks: int,
        stride: int,
    ) -> nn.Sequential:
        """Build one ResNet stage as an nn.Sequential of `n_blocks` residual blocks.

        The first block handles any spatial downsampling (stride argument) and
        any channel dimension change (via a 1×1 projection shortcut).  Subsequent
        blocks always have stride=1 and no shortcut needed.
        """
        downsample = None
        out_channels = planes * block.expansion

        # Need a projection shortcut if spatial resolution or channel count changes
        if stride != 1 or self._in_planes != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self._in_planes, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        blocks = [block(self._in_planes, planes, stride, downsample)]
        self._in_planes = out_channels

        # Subsequent blocks: in_planes == out_channels, stride=1, no downsample
        for _ in range(1, n_blocks):
            blocks.append(block(self._in_planes, planes))

        return nn.Sequential(*blocks)

    def _init_weights(self) -> None:
        """Kaiming normal for all conv layers; constant BN; zero last BN in blocks."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Kaiming normal: calibrates variance for ReLU nonlinearity
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Zero-init the last BN in each residual branch so residual blocks start
        # as near-identity mappings — improves gradient flow at training start
        for m in self.modules():
            if isinstance(m, Bottleneck) and m.bn3 is not None:
                nn.init.zeros_(m.bn3.weight)
            elif isinstance(m, BasicBlock) and m.bn2 is not None:
                nn.init.zeros_(m.bn2.weight)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> BackboneOutput:
        """Run ResNet forward pass and return multi-scale features + pooled embedding.

        Args:
            x: Float32 tensor [B, in_chans, H, W].

        Returns:
            BackboneOutput:
              pooled_output — [B, embed_dim] global representation
              patch_tokens  — None (CNNs produce spatial maps, not tokens)
              feature_maps  — [stage0_out, stage1_out, stage2_out, stage3_out]
                              shapes: [B,64,H/4,W/4], [B,128,H/8,W/8],
                                      [B,256,H/16,W/16], [B,512,H/32,W/32]
                              (channels × 4 for Bottleneck models)
        """
        x = self.stem(x)           # [B, 64, H/4, W/4]

        s0 = self.stage0(x)        # [B, 64  or 256, H/4, W/4]
        s1 = self.stage1(s0)       # [B, 128 or 512, H/8, W/8]
        s2 = self.stage2(s1)       # [B, 256 or 1024, H/16, W/16]
        s3 = self.stage3(s2)       # [B, 512 or 2048, H/32, W/32]

        # Global pooled embedding: flatten spatial dims after AdaptiveAvgPool
        pooled = self.dropout(self.pool(s3).flatten(1))  # [B, embed_dim]

        # Select which scales to return
        all_maps = [s0, s1, s2, s3]
        if self.out_indices is not None:
            all_maps = [all_maps[i] for i in self.out_indices]

        return BackboneOutput(
            pooled_output=pooled,
            patch_tokens=None,    # CNNs have no patch token abstraction
            feature_maps=all_maps,
            attentions=None,
        )

    # ── Stage freezing ────────────────────────────────────────────────────────

    def freeze_up_to_stage(self, stage: int) -> None:
        """Freeze the stem and stages 0 through `stage` (inclusive).

        Useful for fine-tuning: keep early low-level features fixed and only
        update later stages.

        Args:
            stage: 0-based stage index.  E.g. stage=1 freezes stem+stage0+stage1.

        Raises:
            ValueError: if stage is out of range [0, 3].
        """
        if stage < 0 or stage > 3:
            raise ValueError(f"stage must be in [0, 3], got {stage}")

        # Always freeze stem along with stage 0
        for p in self.stem.parameters():
            p.requires_grad_(False)

        for i in range(stage + 1):
            for p in self._stages[i].parameters():
                p.requires_grad_(False)


# ── Registered variants ───────────────────────────────────────────────────────

@register_vision_backbone("resnet18")
class ResNet18(ResNet):
    """ResNet-18: BasicBlock × [2,2,2,2] — lightweight, fast, embed_dim=512."""
    arch_name  = "resnet18"
    embed_dim  = 512
    feature_dims = [64, 128, 256, 512]

    def __init__(self, **kwargs) -> None:
        super().__init__(BasicBlock, [2, 2, 2, 2], **kwargs)


@register_vision_backbone("resnet34")
class ResNet34(ResNet):
    """ResNet-34: BasicBlock × [3,4,6,3] — better accuracy than 18, same embed_dim."""
    arch_name  = "resnet34"
    embed_dim  = 512
    feature_dims = [64, 128, 256, 512]

    def __init__(self, **kwargs) -> None:
        super().__init__(BasicBlock, [3, 4, 6, 3], **kwargs)


@register_vision_backbone("resnet50")
class ResNet50(ResNet):
    """ResNet-50: Bottleneck × [3,4,6,3] — standard production backbone, embed_dim=2048."""
    arch_name  = "resnet50"
    embed_dim  = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__(Bottleneck, [3, 4, 6, 3], **kwargs)


@register_vision_backbone("resnet101")
class ResNet101(ResNet):
    """ResNet-101: Bottleneck × [3,4,23,3] — stronger features, 23 blocks in stage2."""
    arch_name  = "resnet101"
    embed_dim  = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__(Bottleneck, [3, 4, 23, 3], **kwargs)


@register_vision_backbone("resnet152")
class ResNet152(ResNet):
    """ResNet-152: Bottleneck × [3,8,36,3] — deep backbone for high-accuracy tasks."""
    arch_name  = "resnet152"
    embed_dim  = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__(Bottleneck, [3, 8, 36, 3], **kwargs)


@register_vision_backbone("resnet200")
class ResNet200(ResNet):
    """ResNet-200: Bottleneck × [3,24,36,3] — very deep, used in scaling studies."""
    arch_name  = "resnet200"
    embed_dim  = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__(Bottleneck, [3, 24, 36, 3], **kwargs)
