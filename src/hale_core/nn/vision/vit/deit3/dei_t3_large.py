"""``DeiT3Large``: DeiT III-L/16 — 307M params."""

from __future__ import annotations

from hale_core.nn.vision.registry import register_vision_backbone
from hale_core.nn.vision.vit.deit3.dei_t3_backbone import DeiT3Backbone


@register_vision_backbone("deit3_large")
class DeiT3Large(DeiT3Backbone):
    """DeiT III-L/16 — 307M params."""

    arch_name = "deit3_large"
    feature_dims = [1024]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 1024)
        kwargs.setdefault("depth", 24)
        kwargs.setdefault("num_heads", 16)
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("ls_init", 1e-6)
        super().__init__(**kwargs)
