"""``MobileOneBlock``: A single MobileOne block with structural re-parameterisation."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hale_core.nn.vision.cnn.mobile_one._helpers import (
    _ConvBN,
    _fuse_bn,
    _identity_dw_kernel,
    _identity_pw_kernel,
)


class MobileOneBlock(nn.Module):
    """A single MobileOne block with structural re-parameterisation.

    In TRAINING mode: uses k parallel branches per conv.
    In INFERENCE mode (after reparametrize()): single fused conv.

    Architecture:
        input
          ↓ Depthwise (groups=in_channels) with k 3×3 branches
          ↓ + identity branch (if stride=1, same C)
        intermediate
          ↓ Pointwise (1×1) with k branches
        output
        (activation applied after both stages by the caller)

    Args:
        in_channels:  Input channel count.
        out_channels: Output channel count.
        stride:       Spatial stride (1 for same-size, 2 for downsampling).
        num_blocks:   k — number of parallel conv branches per stage.
                      k=1 gives the inference model; k>1 for rich training.
        use_act:      Whether to apply ReLU after this block.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        num_blocks: int = 1,
        use_act: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.num_blocks = num_blocks
        self.use_act = use_act
        self._is_reparam = False  # True after reparametrize()

        # ── Depthwise convolution branches ───────────────────────────────────
        self.dw_branches: nn.ModuleList = nn.ModuleList(
            [
                _ConvBN(in_channels, in_channels, kernel_size=3, stride=stride, groups=in_channels)
                for _ in range(num_blocks)
            ]
        )
        # 1×1 branch in depthwise (acts as identity-like for same-size kernels)
        self.dw_skip_1x1: Optional[_ConvBN] = None
        if in_channels == out_channels:
            self.dw_skip_1x1 = _ConvBN(
                in_channels, in_channels, kernel_size=1, stride=stride, groups=in_channels
            )
        # Identity branch for depthwise (only if stride=1 and same channels)
        self.dw_bn: Optional[nn.BatchNorm2d] = None
        if stride == 1 and in_channels == out_channels:
            self.dw_bn = nn.BatchNorm2d(in_channels)

        # ── Pointwise convolution branches ───────────────────────────────────
        self.pw_branches: nn.ModuleList = nn.ModuleList(
            [_ConvBN(in_channels, out_channels, kernel_size=1) for _ in range(num_blocks)]
        )
        self.pw_bn: Optional[nn.BatchNorm2d] = None
        if in_channels == out_channels:
            # Identity branch for pointwise
            self.pw_bn = nn.BatchNorm2d(in_channels)

        # ── Activation ───────────────────────────────────────────────────────
        self.act = nn.ReLU() if use_act else nn.Identity()

        # ── Reparam'd single conv (filled by reparametrize()) ─────────────────
        self.reparam_dw: Optional[nn.Conv2d] = None
        self.reparam_pw: Optional[nn.Conv2d] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of :class:`MobileOneBlock`.

        Args:
            x: input (``torch.Tensor``).

        Returns:
            ``torch.Tensor``.
        """
        if self._is_reparam:
            # Fast inference path: single fused conv
            return self.act(self.reparam_pw(self.act(self.reparam_dw(x))))

        # ── Depthwise stage ───────────────────────────────────────────────────
        dw_out = sum(branch(x) for branch in self.dw_branches)
        if self.dw_skip_1x1 is not None:
            dw_out = dw_out + self.dw_skip_1x1(x)
        if self.dw_bn is not None:
            dw_out = dw_out + self.dw_bn(x)  # identity residual (BN only)
        dw_out = self.act(dw_out)

        # ── Pointwise stage ───────────────────────────────────────────────────
        pw_out = sum(branch(dw_out) for branch in self.pw_branches)
        if self.pw_bn is not None:
            pw_out = pw_out + self.pw_bn(dw_out)  # identity residual
        return self.act(pw_out)

    def reparametrize(self) -> None:
        """Fuse all parallel branches into a single conv+bias each.

        After calling this:
          - All branch parameters are deleted (memory freed).
          - self.reparam_dw / self.reparam_pw hold the fused convs.
          - forward() uses the fast inference path.
        """
        if self._is_reparam:
            return

        # ── Fuse depthwise ────────────────────────────────────────────────────
        dw_w = torch.zeros_like(self.dw_branches[0][0].weight)  # [C, 1, 3, 3] for groups=C
        dw_b = torch.zeros(self.in_channels, device=dw_w.device, dtype=dw_w.dtype)

        for branch in self.dw_branches:
            w, b = _fuse_bn(branch[0], branch[1])
            dw_w = dw_w + w
            dw_b = dw_b + b

        if self.dw_skip_1x1 is not None:
            w, b = _fuse_bn(self.dw_skip_1x1[0], self.dw_skip_1x1[1])
            # Pad 1×1 weight to 3×3 (centre) to match branch kernels
            w_padded = F.pad(w, [1, 1, 1, 1])
            dw_w = dw_w + w_padded
            dw_b = dw_b + b

        if self.dw_bn is not None:
            # Identity branch: weight is an identity 3×3 depthwise kernel
            eye_w = _identity_dw_kernel(self.in_channels, device=dw_w.device, dtype=dw_w.dtype)
            scale = self.dw_bn.weight / (self.dw_bn.running_var + self.dw_bn.eps).sqrt()
            bias = self.dw_bn.bias - self.dw_bn.running_mean * scale
            dw_w = dw_w + eye_w * scale.view(-1, 1, 1, 1)
            dw_b = dw_b + bias

        self.reparam_dw = nn.Conv2d(
            self.in_channels,
            self.in_channels,
            kernel_size=3,
            stride=self.stride,
            padding=1,
            groups=self.in_channels,
            bias=True,
        )
        self.reparam_dw.weight.data = dw_w
        self.reparam_dw.bias.data = dw_b

        # ── Fuse pointwise ────────────────────────────────────────────────────
        pw_w = torch.zeros_like(self.pw_branches[0][0].weight)  # [C_out, C_in, 1, 1]
        pw_b = torch.zeros(self.out_channels, device=pw_w.device, dtype=pw_w.dtype)

        for branch in self.pw_branches:
            w, b = _fuse_bn(branch[0], branch[1])
            pw_w = pw_w + w
            pw_b = pw_b + b

        if self.pw_bn is not None and self.in_channels == self.out_channels:
            # Identity branch: weight is 1×1 identity (eye) for pointwise
            eye_w = _identity_pw_kernel(self.in_channels, device=pw_w.device, dtype=pw_w.dtype)
            scale = self.pw_bn.weight / (self.pw_bn.running_var + self.pw_bn.eps).sqrt()
            bias = self.pw_bn.bias - self.pw_bn.running_mean * scale
            pw_w = pw_w + eye_w * scale.view(-1, 1, 1, 1)
            pw_b = pw_b + bias

        self.reparam_pw = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=1,
            bias=True,
        )
        self.reparam_pw.weight.data = pw_w
        self.reparam_pw.bias.data = pw_b

        # ── Delete training-time branches (free memory) ───────────────────────
        del self.dw_branches, self.dw_skip_1x1, self.dw_bn
        del self.pw_branches, self.pw_bn
        self._is_reparam = True
