"""``DeiT3Base``: DeiT III-B/16 — 86M params."""

from __future__ import annotations

from hale_core.nn.vision.registry import register_vision_backbone
from hale_core.nn.vision.vit.deit3.dei_t3_backbone import DeiT3Backbone


@register_vision_backbone("deit3_base")
class DeiT3Base(DeiT3Backbone):
    """DeiT III-B/16 — 86M params."""

    arch_name = "deit3_base"
    feature_dims = [768]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 768)
        kwargs.setdefault("depth", 12)
        kwargs.setdefault("num_heads", 12)
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("ls_init", 1e-6)
        super().__init__(**kwargs)
