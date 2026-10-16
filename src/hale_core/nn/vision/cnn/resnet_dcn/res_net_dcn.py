"""``ResNetDCN``: ResNet backbone where stages 2 and 3 use Deformable Convolution v2."""

from __future__ import annotations

from hale_core.nn.vision.cnn.resnet import Bottleneck, ResNet
from hale_core.nn.vision.cnn.resnet_dcn.dcn_bottleneck import DCNBottleneck


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
        stage_planes = [64, 128, 256, 512]
        stage_strides = [1, 2, 2, 2]
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
