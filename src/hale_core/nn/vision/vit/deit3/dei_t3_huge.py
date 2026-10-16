"""``DeiT3Huge``: DeiT III-H/14 — 632M params.  The flagship DeiT III model."""

from __future__ import annotations

from hale_core.nn.vision.registry import register_vision_backbone
from hale_core.nn.vision.vit.deit3.dei_t3_backbone import DeiT3Backbone


@register_vision_backbone("deit3_huge")
class DeiT3Huge(DeiT3Backbone):
    """DeiT III-H/14 — 632M params.  The flagship DeiT III model."""

    arch_name = "deit3_huge"
    feature_dims = [1280]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 1280)
        kwargs.setdefault("depth", 32)
        kwargs.setdefault("num_heads", 16)
        kwargs.setdefault("patch_size", 14)
        kwargs.setdefault("ls_init", 1e-6)
        super().__init__(**kwargs)
