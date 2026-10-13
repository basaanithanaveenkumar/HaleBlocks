"""
ResNet + Deformable Convolution v2 (DCNv2) Backbones
=====================================================

Extends ResNet-50/101 by replacing the 3×3 convolutions in late stages
(stages 2 and 3) with Deformable Convolution v2 (Zhu et al. 2019).

Why deformable convolutions?
-----------------------------
Standard convolutions sample a fixed rectangular grid of input positions,
which is inefficient for objects at non-canonical scales, poses, or
orientations.  Deformable convolutions learn *per-position offsets* that
shift each sampling point, allowing the receptive field to adapt to the
geometry of the image content.

DCNv2 vs DCNv1
--------------
DCNv1 (Dai et al. 2017): learns 2D offsets per sampling point (Δx, Δy for
  each of the k² kernel positions).
DCNv2 (Zhu et al. 2019): additionally learns a *modulation scalar* per
  sampling point (a sigmoid gating weight in [0, 1]) that controls how much
  each sampled position contributes.  This gives the network more flexibility
  to suppress irrelevant context.

Implementation
--------------
This module implements DCNv2 from scratch using torch grid_sample, avoiding
the need for CUDA-compiled extensions like mmcv/torchvision deform_conv.

For each spatial location (i, j) and kernel position (p, q):
  1. An offset_conv (small 3×3 conv → 2 × k² channels) produces Δx, Δy.
  2. A mask_conv   (small 3×3 conv → k² channels)       produces weights ∈ (0,1).
  3. The sampled features are: sum over (p,q) of mask[p,q] * sample(x, i+p+Δx, j+q+Δy)
  4. Sampling uses bilinear interpolation via F.grid_sample.

The offset_conv is zero-initialized so the network starts as a regular conv
and learns deformability over time.

Stages using DCN
----------------
  stage2 (stride-16): 3×3 convolutions replaced with DeformConv2d
  stage3 (stride-32): 3×3 convolutions replaced with DeformConv2d

Stage 0 (stride-4) and stage 1 (stride-8) retain standard convolutions
because:
  - Low-level texture features are well-captured by regular grids.
  - DCN introduces more parameters and compute; keeping it to later stages
    is a good FLOPs/accuracy tradeoff.

Registered names
----------------
  resnet50_dcn  — ResNet-50 with DCNv2 in stages 2 & 3
  resnet101_dcn — ResNet-101 with DCNv2 in stages 2 & 3

Citation
--------
  Dai et al. (2017). Deformable Convolutional Networks. ICCV.
  Zhu et al. (2019). Deformable ConvNets v2: More Deformable, Better Results. CVPR.
"""

from __future__ import annotations

from typing import Optional, Type

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.base import BackboneOutput, BaseVisionBackbone
from hale_core.nn.vision.cnn.resnet import BasicBlock, Bottleneck, ResNet
from hale_core.nn.vision.registry import register_vision_backbone


# ── Deformable Convolution v2 ────────────────────────────────────────────────

class DeformConv2d(nn.Module):
    """Deformable Convolution v2 implemented via grid_sample.

    For each output spatial location, this layer:
      1. Predicts 2*k*k offsets (Δy, Δx per kernel grid point) via `offset_conv`.
      2. Predicts k*k modulation weights (sigmoid) via `mask_conv`.
      3. Samples input at offset positions using bilinear interpolation.
      4. Applies the learned kernel weights to the sampled features.

    The implementation uses F.grid_sample which handles out-of-bounds inputs
    by zero-padding, matching the typical CUDA DCN extension behaviour.

    Args:
        in_channels:  Number of input feature map channels.
        out_channels: Number of output feature map channels.
        kernel_size:  Spatial kernel size (typically 3).
        stride:       Stride of the convolution.
        padding:      Zero-padding applied to input (typically kernel_size//2).
        bias:         Whether to add a bias term to the output.

    Shape:
        Input:  [B, in_channels, H, W]
        Output: [B, out_channels, H', W']  where H', W' depend on stride/padding.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int  = 3,
        stride:       int  = 1,
        padding:      int  = 1,
        bias:         bool = False,
    ) -> None:
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.stride       = stride
        self.padding      = padding
        self.k2 = kernel_size * kernel_size  # number of kernel grid positions

        # Offset predictor: 2 × k² channels (Δy and Δx for each grid point)
        # Zero-initialized so the deformation starts as identity (regular conv)
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * self.k2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)

        # Modulation predictor: k² channels per spatial location, sigmoid → [0,1]
        self.mask_conv = nn.Conv2d(
            in_channels,
            self.k2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True,
        )
        # Init mask weights to small values so masks start near 0.5 (sigmoid(0)=0.5)
        nn.init.zeros_(self.mask_conv.weight)
        nn.init.zeros_(self.mask_conv.bias)

        # Main convolution weight (applied to sampled features)
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias_param = nn.Parameter(torch.zeros(out_channels)) if bias else None

        # Standard Kaiming init for conv weights
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Deformable conv2d forward pass.

        Strategy:
          - Predict per-location offsets and masks from input x.
          - Build a sampling grid of shape [B, H_out*W_out*k², 2] by adding
            offsets to the base regular grid.
          - Use grid_sample to gather bilinearly-interpolated samples.
          - Apply modulation weights and sum over kernel positions.
          - Finally apply the weight matrix to produce output channels.
        """
        B, C, H, W = x.shape

        # Predict offsets: [B, 2*k², H_out, W_out]
        offsets = self.offset_conv(x)
        # Predict masks: [B, k², H_out, W_out]
        masks = torch.sigmoid(self.mask_conv(x))

        H_out = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        W_out = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        # ── Build base regular-grid sampling coordinates ──────────────────────
        # Output positions in input-pixel space, accounting for stride
        out_y = torch.arange(H_out, device=x.device, dtype=x.dtype) * self.stride - self.padding
        out_x = torch.arange(W_out, device=x.device, dtype=x.dtype) * self.stride - self.padding

        # Kernel offsets relative to center (e.g. for k=3: -1, 0, +1 in each dim)
        ky_range = torch.arange(self.kernel_size, device=x.device, dtype=x.dtype) - self.kernel_size // 2
        kx_range = torch.arange(self.kernel_size, device=x.device, dtype=x.dtype) - self.kernel_size // 2

        # Base grid: [H_out, W_out, k, k, 2]  (y and x for each kernel position)
        base_y = out_y.view(H_out, 1, 1, 1) + ky_range.view(1, 1, self.kernel_size, 1)
        base_x = out_x.view(1, W_out, 1, 1) + kx_range.view(1, 1, 1, self.kernel_size)

        # Reshape to [H_out, W_out, k²]
        base_y = base_y.expand(H_out, W_out, self.kernel_size, self.kernel_size).reshape(H_out, W_out, self.k2)
        base_x = base_x.expand(H_out, W_out, self.kernel_size, self.kernel_size).reshape(H_out, W_out, self.k2)

        # ── Add learned offsets ───────────────────────────────────────────────
        # offsets shape: [B, 2*k², H_out, W_out] → split into dy, dx
        dy = offsets[:, :self.k2, :, :].permute(0, 2, 3, 1)   # [B, H_out, W_out, k²]
        dx = offsets[:, self.k2:, :, :].permute(0, 2, 3, 1)   # [B, H_out, W_out, k²]

        # Absolute sampling positions in input-pixel space
        sample_y = base_y.unsqueeze(0) + dy   # [B, H_out, W_out, k²]
        sample_x = base_x.unsqueeze(0) + dx   # [B, H_out, W_out, k²]

        # ── Normalise coordinates to [-1, 1] for grid_sample ─────────────────
        # grid_sample expects x,y in [-1, 1] where (-1,-1) = top-left corner
        norm_y = 2.0 * sample_y / (H - 1) - 1.0
        norm_x = 2.0 * sample_x / (W - 1) - 1.0

        # Stack into grid: [B, H_out, W_out*k², 2] (x first for grid_sample)
        # We sample all H_out*W_out*k² points in one grid_sample call
        grid = torch.stack([norm_x, norm_y], dim=-1)              # [B, H_out, W_out, k², 2]
        grid = grid.view(B, H_out, W_out * self.k2, 2)            # [B, H_out, W_out*k², 2]

        # ── Bilinear sampling ─────────────────────────────────────────────────
        # x: [B, C, H, W] → sample at grid positions → [B, C, H_out, W_out*k²]
        sampled = F.grid_sample(x, grid, mode="bilinear", align_corners=True, padding_mode="zeros")
        sampled = sampled.view(B, C, H_out, W_out, self.k2)       # [B, C, H_out, W_out, k²]
        sampled = sampled.permute(0, 2, 3, 4, 1)                  # [B, H_out, W_out, k², C]

        # ── Apply modulation masks ────────────────────────────────────────────
        masks = masks.permute(0, 2, 3, 1).unsqueeze(-1)           # [B, H_out, W_out, k², 1]
        sampled = sampled * masks                                   # broadcast over C

        # ── Apply conv weights (einsum over k² kernel positions and C) ────────
        # weight: [out_channels, C, k, k] → reshape to [out_channels, C, k²]
        w = self.weight.view(self.out_channels, C, self.k2)        # [O, C, k²]

        # sampled: [B, H_out, W_out, k², C] → sum over k² and C
        # out[b,h,w,o] = sum_{p,c} sampled[b,h,w,p,c] * w[o,c,p]
        out = torch.einsum("bhwpc,ocp->bhwo", sampled, w)          # [B, H_out, W_out, O]
        out = out.permute(0, 3, 1, 2)                              # [B, O, H_out, W_out]

        if self.bias_param is not None:
            out = out + self.bias_param.view(1, -1, 1, 1)

        return out


# ── DCN Bottleneck block ──────────────────────────────────────────────────────

class DCNBottleneck(nn.Module):
    """Bottleneck block with DeformConv2d replacing the 3×3 standard conv.

    Identical to Bottleneck except conv2 is a DeformConv2d.  The 1×1 squeeze
    and 1×1 expand convolutions remain standard — they operate on single
    spatial locations and so don't benefit from deformability.

    Args:
        in_planes:  Input channel count.
        planes:     Bottleneck width (output = planes × 4).
        stride:     Spatial stride (downsampling) for the deformable 3×3 conv.
        downsample: Optional projection shortcut.
    """

    expansion: int = 4

    def __init__(
        self,
        in_planes:  int,
        planes:     int,
        stride:     int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        out_channels = planes * self.expansion

        # 1×1 squeeze
        self.conv1 = nn.Conv2d(in_planes, planes, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)

        # DCNv2 replaces the 3×3 standard conv
        self.conv2 = DeformConv2d(planes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn2   = nn.BatchNorm2d(planes)

        # 1×1 expand
        self.conv3 = nn.Conv2d(planes, out_channels, 1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        return F.relu(out + identity, inplace=True)


# ── ResNet-DCN backbone ───────────────────────────────────────────────────────

class ResNetDCN(ResNet):
    """ResNet backbone where stages 2 and 3 use Deformable Convolution v2.

    Inherits the full ResNet logic; only the block class used for stages 2/3
    is overridden.  Stages 0 and 1 use standard Bottleneck blocks.

    The stage split is controlled by `dcn_stages` (default: [2, 3]).

    Args:
        layers:     Number of blocks per stage (same as standard ResNet).
        dcn_stages: Which stage indices (0-based) use DCNBottleneck.
        **kwargs:   Forwarded to ResNet.__init__.
    """

    def __init__(
        self,
        layers: list[int],
        dcn_stages: list[int] = None,
        **kwargs,
    ) -> None:
        if dcn_stages is None:
            dcn_stages = [2, 3]

        # Call ResNet init with standard Bottleneck first (builds all 4 stages)
        super().__init__(Bottleneck, layers, **kwargs)

        # Now rebuild stages that should use DCN blocks, preserving channel counts
        # We must manually track _in_planes because ResNet.__init__ has already advanced it
        self._dcn_stages = dcn_stages

        # Rebuild stages 0–3 with correct block types
        # Stage planes and initial in_planes for ResNet with Bottleneck expansion=4
        stage_planes   = [64, 128, 256, 512]
        stage_strides  = [1,  2,   2,   2]
        in_planes_list = [64, 256, 512, 1024]  # input channels entering each stage

        for i, (planes, stride, in_planes) in enumerate(
            zip(stage_planes, stage_strides, in_planes_list)
        ):
            block_cls = DCNBottleneck if i in dcn_stages else Bottleneck
            self._in_planes = in_planes
            stage = self._make_stage(block_cls, planes, layers[i], stride)
            setattr(self, f"stage{i}", stage)

        # Update _stages list reference so freeze_up_to_stage works
        self._stages = [self.stage0, self.stage1, self.stage2, self.stage3]

        # Reinit weights to catch any newly created DCN blocks
        self._init_weights()


@register_vision_backbone("resnet50_dcn")
class ResNet50DCN(ResNetDCN):
    """ResNet-50 with DeformConv v2 in stages 2 & 3 — strong detector backbone."""
    arch_name    = "resnet50_dcn"
    embed_dim    = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__([3, 4, 6, 3], dcn_stages=[2, 3], **kwargs)


@register_vision_backbone("resnet101_dcn")
class ResNet101DCN(ResNetDCN):
    """ResNet-101 with DeformConv v2 in stages 2 & 3 — high-capacity detector backbone."""
    arch_name    = "resnet101_dcn"
    embed_dim    = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__([3, 4, 23, 3], dcn_stages=[2, 3], **kwargs)
