"""
Vision Backbone Registry
=========================

Central registry for all vision backbone implementations.

Design
------
Two levels of registration:
  1. VISION_BACKBONES  — maps "name" → backbone class
  2. BACKBONE_CONFIGS  — maps "name" → list of available config variants
                          e.g. "resnet" → ["resnet18", "resnet50", "resnet101"]

Usage
-----
Register a backbone (done automatically when the module is imported):

    from hale_core.nn.vision.registry import register_vision_backbone

    @register_vision_backbone("resnet50")
    class ResNet50(BaseVisionBackbone):
        ...

Build a backbone from a name string (used by VLM, WAM, Halos-lift):

    from hale_core.nn.vision.registry import build_vision_backbone

    backbone = build_vision_backbone(
        "convnext_base",
        img_size=224,
        drop_rate=0.1,
    )

List all registered backbones:

    from hale_core.nn.vision.registry import list_vision_backbones
    print(list_vision_backbones())
    # ["resnet18", "resnet34", ..., "convnext_base", ..., "vit_large", ...]
"""

from __future__ import annotations

from typing import Any, TypeVar

from loguru import logger

from hale_core.nn.vision.base import BaseVisionBackbone

T = TypeVar("T", bound=type[BaseVisionBackbone])

# ── Registry storage ─────────────────────────────────────────────────────────
# Maps lowercase name → backbone class.
# Populated by @register_vision_backbone decorators at module import time.
_VISION_BACKBONES: dict[str, type[BaseVisionBackbone]] = {}


def register_vision_backbone(name: str) -> "Callable[[T], T]":
    """Class decorator that registers a backbone under the given name.

    Args:
        name: Unique string key (lowercase, snake_case).
              Convention: "<family>_<size>" e.g. "resnet50", "vit_large",
              "convnext_base", "swinv2_tiny".

    Returns:
        The unmodified class (decorator is side-effect-only).

    Raises:
        ValueError if the name is already registered.

    Example::

        @register_vision_backbone("my_backbone")
        class MyBackbone(BaseVisionBackbone):
            arch_name = "my_backbone"
            embed_dim = 512
    """
    def decorator(cls: T) -> T:
        key = name.lower()
        if key in _VISION_BACKBONES:
            raise ValueError(
                f"Vision backbone '{key}' is already registered "
                f"(existing: {_VISION_BACKBONES[key].__name__})"
            )
        _VISION_BACKBONES[key] = cls
        logger.debug("registered vision backbone '{}'", key)
        return cls
    return decorator


def build_vision_backbone(name: str, **kwargs: Any) -> BaseVisionBackbone:
    """Instantiate a registered vision backbone by name.

    Args:
        name:    Backbone name as registered (case-insensitive).
        **kwargs: Constructor arguments forwarded to the backbone class.
                  Common args: img_size, in_chans, drop_rate, out_indices.

    Returns:
        An instance of the registered backbone class.

    Raises:
        KeyError if the name is not registered.
        TypeError if kwargs don't match the constructor.

    Example::

        backbone = build_vision_backbone("vit_large", img_size=336)
        out = backbone(images)   # BackboneOutput
    """
    _ensure_backbones_imported()
    key = name.lower()
    if key not in _VISION_BACKBONES:
        available = sorted(_VISION_BACKBONES.keys())
        raise KeyError(
            f"Unknown vision backbone '{name}'. "
            f"Available: {available}"
        )
    cls = _VISION_BACKBONES[key]
    return cls(**kwargs)


def get_vision_backbone_cls(name: str) -> type[BaseVisionBackbone]:
    """Return the class (not an instance) for a registered backbone name."""
    _ensure_backbones_imported()
    key = name.lower()
    if key not in _VISION_BACKBONES:
        raise KeyError(f"Unknown vision backbone '{name}'")
    return _VISION_BACKBONES[key]


def list_vision_backbones() -> list[str]:
    """Return a sorted list of all registered backbone names."""
    _ensure_backbones_imported()
    return sorted(_VISION_BACKBONES.keys())


def list_vision_backbones_by_family() -> dict[str, list[str]]:
    """Group registered backbones by family prefix for display.

    Returns a dict like:
        {
            "resnet": ["resnet18", "resnet34", "resnet50", ...],
            "vit":    ["vit_base", "vit_large", "vit_huge", ...],
            ...
        }
    """
    _ensure_backbones_imported()
    families: dict[str, list[str]] = {}
    for name in sorted(_VISION_BACKBONES.keys()):
        # Family is everything before the first underscore
        parts = name.split("_")
        family = parts[0]
        families.setdefault(family, []).append(name)
    return families


# ── Lazy import trigger ───────────────────────────────────────────────────────
_imported = False

def _ensure_backbones_imported() -> None:
    """Import all backbone sub-packages so their @register decorators run.

    This is called lazily on the first build/list call, avoiding circular
    imports at package load time.
    """
    global _imported
    if _imported:
        return
    _imported = True

    # Each import triggers @register_vision_backbone decorators in that module
    # CNN Era (2012–2020)
    from hale_core.nn.vision.cnn import alexnet        # noqa: F401
    from hale_core.nn.vision.cnn import vgg            # noqa: F401
    from hale_core.nn.vision.cnn import inception      # noqa: F401
    from hale_core.nn.vision.cnn import resnet         # noqa: F401
    from hale_core.nn.vision.cnn import resnet_dcn     # noqa: F401
    from hale_core.nn.vision.cnn import resnext        # noqa: F401
    from hale_core.nn.vision.cnn import densenet       # noqa: F401
    from hale_core.nn.vision.cnn import mobilenet_v1   # noqa: F401
    from hale_core.nn.vision.cnn import senet          # noqa: F401
    from hale_core.nn.vision.cnn import efficientnet_b # noqa: F401
    from hale_core.nn.vision.cnn import resnest        # noqa: F401
    # Transition Era (2020–2021)
    from hale_core.nn.vision.cnn import nfnet          # noqa: F401
    from hale_core.nn.vision.cnn import regnet         # noqa: F401
    from hale_core.nn.vision.cnn import convnext_v1    # noqa: F401
    # Modern CNN Era
    from hale_core.nn.vision.cnn import convnext       # noqa: F401
    from hale_core.nn.vision.cnn import mobile_one     # noqa: F401
    from hale_core.nn.vision.cnn import efficientnet   # noqa: F401
    from hale_core.nn.vision.cnn import internimage    # noqa: F401
    # Transformer Era
    from hale_core.nn.vision.vit import vit            # noqa: F401
    from hale_core.nn.vision.vit import dino           # noqa: F401
    from hale_core.nn.vision.vit import eva            # noqa: F401
    from hale_core.nn.vision.vit import deit3          # noqa: F401
    from hale_core.nn.vision.swin import swinv2        # noqa: F401
    from hale_core.nn.vision.hybrid import coatnet     # noqa: F401
    from hale_core.nn.vision.hybrid import maxvit      # noqa: F401
