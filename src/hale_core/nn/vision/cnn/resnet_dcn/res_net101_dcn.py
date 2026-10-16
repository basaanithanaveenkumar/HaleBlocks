"""``ResNet101DCN``: ResNet-101 with DeformConv v2 in stages 2 & 3 — high-capacity detector backbone."""

from __future__ import annotations

from hale_core.nn.vision.cnn.resnet_dcn.res_net_dcn import ResNetDCN
from hale_core.nn.vision.registry import register_vision_backbone


@register_vision_backbone("resnet101_dcn")
class ResNet101DCN(ResNetDCN):
    """ResNet-101 with DeformConv v2 in stages 2 & 3 — high-capacity detector backbone."""

    arch_name = "resnet101_dcn"
    embed_dim = 2048
    feature_dims = [256, 512, 1024, 2048]

    def __init__(self, **kwargs) -> None:
        super().__init__([3, 4, 23, 3], dcn_stages=[2, 3], **kwargs)
