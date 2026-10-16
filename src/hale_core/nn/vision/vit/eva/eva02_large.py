"""``EVA02Large``: EVA-02-L/14 — 307M params.  The flagship EVA-02 model."""

from __future__ import annotations

from hale_core.nn.vision.registry import register_vision_backbone
from hale_core.nn.vision.vit.eva.eva02_backbone import EVA02Backbone


@register_vision_backbone("eva_02_large")
class EVA02Large(EVA02Backbone):
    """EVA-02-L/14 — 307M params.  The flagship EVA-02 model."""

    arch_name = "eva_02_large"
    feature_dims = [1024]

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("embed_dim", 1024)
        kwargs.setdefault("depth", 24)
        kwargs.setdefault("num_heads", 16)
        kwargs.setdefault("patch_size", 14)
        super().__init__(**kwargs)
