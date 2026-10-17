"""hale-blocks: reusable transformer core, config, training, and plugin registries.

Top-level names are resolved lazily (PEP 562): ``import hale_core`` costs almost nothing, and
torch, the layers and the registries load on first use of a name below, or when a subpackage
such as :mod:`hale_core.registry` is imported directly. Registration side effects are
unchanged, because they live in the subpackages (``hale_core.registry`` registers the
built-ins when *it* is imported), not in this file.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

#: public name -> subpackage that defines it
_EXPORTS = {
    "BACKBONES": "hale_core.nn",
    "DiTBackbone": "hale_core.nn",
    "DiTStack": "hale_core.nn",
    "GPTBackbone": "hale_core.nn",
    "GPTStack": "hale_core.nn",
    "LGTBackbone": "hale_core.nn",
    "LGTStack": "hale_core.nn",
    "TransformerBackbone": "hale_core.nn",
    "build_attn_mask": "hale_core.nn",
    "build_backbone": "hale_core.nn",
    "NamedRegistry": "hale_core.registry",
    "VariantRegistry": "hale_core.registry",
    "get_loss": "hale_core.registry",
    "get_model": "hale_core.registry",
    "get_optimizer": "hale_core.registry",
    "get_sampler": "hale_core.registry",
    "get_variant": "hale_core.registry",
    "CheckpointStore": "hale_core.runtime",
    "load_checkpoint": "hale_core.runtime",
    "save_checkpoint": "hale_core.runtime",
}

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name: str) -> Any:
    """Import ``name`` from its subpackage on first access and cache it on the module."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'hale_core' has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(__all__)
