"""Shared helpers of :mod:`hale_core.nn.vision.cnn.vgg` (VGGNet Vision Backbones)."""

from __future__ import annotations

import torch.nn as nn

# ── VGG configuration table ───────────────────────────────────────────────────
# Each list entry is either an int (out_channels for a Conv3×3) or 'M' (MaxPool).
# Five MaxPools divide the backbone into 5 stages.
_CFGS: dict[str, list[int | str]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "vgg19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
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
