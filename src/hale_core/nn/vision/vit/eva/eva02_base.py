"""``EVA02Base``: EVA-02-B/16 — 86M params."""

from __future__ import annotations

from hale_core.nn.vision.registry import register_vision_backbone
from hale_core.nn.vision.vit.eva.eva02_backbone import EVA02Backbone


@register_vision_backbone("eva_02_base")
class EVA02Base(EVA02Backbone):
    """EVA-02-B/16 — 86M params."""

    arch_name = "eva_02_base"
    feature_dims = [768]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 768)
        kwargs.setdefault("depth", 12)
        kwargs.setdefault("num_heads", 12)
        kwargs.setdefault("patch_size", 16)
        super().__init__(**kwargs)
