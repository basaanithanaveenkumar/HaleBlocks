"""``ViT22B``: ViT-22B/14 — 22B params (Zhai et al., 2022)."""

from __future__ import annotations

from hale_core.nn.vision.registry import register_vision_backbone
from hale_core.nn.vision.vit.vit.vi_t_backbone import ViTBackbone


@register_vision_backbone("vit_22b")
class ViT22B(ViTBackbone):
    """ViT-22B/14 — 22B params (Zhai et al., 2022).

    At this scale the model is expected to be sharded across devices.
    The class provides the architecture; distributed loading is caller's responsibility.

    Reference: "Scaling Vision Transformers to 22 Billion Parameters" (Zhai 2022)
               https://arxiv.org/abs/2302.05442
    """

    arch_name = "vit_22b"
    feature_dims = [6144]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 6144)
        kwargs.setdefault("depth", 48)
        kwargs.setdefault("num_heads", 48)
        kwargs.setdefault("patch_size", 14)
        super().__init__(**kwargs)
