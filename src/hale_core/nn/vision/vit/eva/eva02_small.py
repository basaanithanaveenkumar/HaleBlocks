"""``EVA02Small``: EVA-02-S/16 — 22M params."""

from __future__ import annotations

from hale_core.nn.vision.registry import register_vision_backbone
from hale_core.nn.vision.vit.eva.eva02_backbone import EVA02Backbone


@register_vision_backbone("eva_02_small")
class EVA02Small(EVA02Backbone):
    """EVA-02-S/16 — 22M params."""

    arch_name = "eva_02_small"
    feature_dims = [384]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 384)
        kwargs.setdefault("depth", 12)
        kwargs.setdefault("num_heads", 6)
        kwargs.setdefault("patch_size", 16)
        super().__init__(**kwargs)
