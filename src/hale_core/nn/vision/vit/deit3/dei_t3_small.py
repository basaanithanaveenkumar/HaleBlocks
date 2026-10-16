"""``DeiT3Small``: DeiT III-S/16 — 22M params."""

from __future__ import annotations

from hale_core.nn.vision.registry import register_vision_backbone
from hale_core.nn.vision.vit.deit3.dei_t3_backbone import DeiT3Backbone


@register_vision_backbone("deit3_small")
class DeiT3Small(DeiT3Backbone):
    """DeiT III-S/16 — 22M params."""

    arch_name = "deit3_small"
    feature_dims = [384]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 384)
        kwargs.setdefault("depth", 12)
        kwargs.setdefault("num_heads", 6)
        kwargs.setdefault("patch_size", 16)
        kwargs.setdefault("ls_init", 1e-6)
        super().__init__(**kwargs)
