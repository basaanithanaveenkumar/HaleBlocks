"""Decorator-based plugin registries.

Registry primitives:
  - ``Registry``: mmdet-style name → class table with ``register_module``, ``build(cfg)``,
    parent/child scopes (the canonical implementation; downstream packages build on it)
  - ``NamedRegistry``: ``Registry`` with the original ``add``/``items`` interface
  - ``VariantRegistry``: one name → many objects, resolved by training variant

Registries exposed here:
  Training-level:  MODELS, LOSSES, SAMPLERS, OPTIMIZERS, CONFIGS, LOGGERS,
                   TRAINERS, DATASETS, METRICS
  NN-component:    BLOCKS, NORMS, COLLATORS, DATALOADERS, SCHEDULERS
  Local (nn pkg):  ATTENTION (attention/factory), FFN (ffn/factory),
                   BACKBONES (backbones/factory) — accessed via lazy helpers below

Config-driven builders (from registry.nn):
  build_attention_from_cfg, build_ffn_from_cfg, build_norm_from_cfg,
  build_block_from_cfg, build_backbone_from_cfg, build_scheduler, build_dataloader
"""

from hale_core.registry.base import NamedRegistry
from hale_core.registry.registry import Registry, build_from_cfg
from hale_core.registry.plugins import (
    # Registries
    BLOCKS,
    BLOCKS_REGISTRY,
    COLLATORS,
    COLLATORS_REGISTRY,
    CONFIGS,
    CONFIGS_REGISTRY,
    DATASETS,
    DATASETS_REGISTRY,
    DATALOADERS,
    DATALOADERS_REGISTRY,
    LOGGERS,
    LOGGERS_REGISTRY,
    LOSSES,
    LOSSES_REGISTRY,
    METRICS,
    METRICS_REGISTRY,
    MODELS,
    MODELS_REGISTRY,
    NORMS,
    NORMS_REGISTRY,
    OPTIMIZERS,
    OPTIMIZERS_REGISTRY,
    SAMPLERS,
    SAMPLERS_REGISTRY,
    SCHEDULERS,
    SCHEDULERS_REGISTRY,
    TRAINERS,
    TRAINERS_REGISTRY,
    VARIANTS,
    # Decorators / getters
    build_logger,
    build_norm,
    get_block,
    get_collator,
    get_config_schema,
    get_dataloader_factory,
    get_dataset,
    get_loss,
    get_model,
    get_norm,
    get_optimizer,
    get_sampler,
    get_scheduler_factory,
    get_trainer,
    get_variant,
    instantiate_metrics,
    list_datasets,
    register_block,
    register_collator,
    register_config,
    register_dataloader,
    register_dataset,
    register_logger,
    register_loss,
    register_metric,
    register_model,
    register_norm,
    register_optimizer,
    register_sampler,
    register_scheduler,
    register_trainer,
    register_variant,
)
from hale_core.registry.variant import VariantRegistry
from hale_core.registry.bootstrap import register_builtins

register_builtins()

# Config-driven builders — imported lazily inside functions; re-exported here
# for a single import surface.
from hale_core.registry.nn import (  # noqa: E402
    build_attention_from_cfg,
    build_backbone_from_cfg,
    build_block_from_cfg,
    build_dataloader,
    build_ffn_from_cfg,
    build_norm_from_cfg,
    build_scheduler,
)


# Lazy accessors for nn-local registries (avoid circular import at module level)
def get_attention_registry():
    """Return the ``ATTENTION`` NamedRegistry from :mod:`hale_core.nn.layers.attention`."""
    from hale_core.nn.layers.attention.factory import ATTENTION
    return ATTENTION


def get_ffn_registry():
    """Return the ``FFN`` NamedRegistry from :mod:`hale_core.nn.layers.ffn`."""
    from hale_core.nn.layers.ffn.factory import FFN
    return FFN


def get_backbone_registry():
    """Return the ``BACKBONES`` NamedRegistry from :mod:`hale_core.nn.backbones`."""
    from hale_core.nn.backbones.factory import BACKBONES
    return BACKBONES


# ── Vision backbone public surface ────────────────────────────────────────────
# These are the production entry points used by Hale-VLM, HALE-WAM, Halos-lift.

def build_vision_backbone(name: str, **kwargs):
    """Build a registered vision backbone by name.

    Args:
        name:    Backbone name, e.g. "convnext_base", "vit_large", "swinv2_tiny".
        **kwargs: Forwarded to the backbone constructor.
                  Common: img_size=224, in_chans=3, drop_rate=0.0.

    Returns:
        An instance of BaseVisionBackbone.

    Example::

        from hale_core.registry import build_vision_backbone

        backbone = build_vision_backbone("convnext_base", img_size=224)
        out = backbone(images)   # BackboneOutput
    """
    from hale_core.nn.vision.registry import build_vision_backbone as _build
    return _build(name, **kwargs)


def list_vision_backbones() -> list[str]:
    """Return all registered vision backbone names."""
    from hale_core.nn.vision.registry import list_vision_backbones as _list
    return _list()


def list_vision_backbones_by_family() -> dict[str, list[str]]:
    """Return registered backbones grouped by family prefix."""
    from hale_core.nn.vision.registry import list_vision_backbones_by_family as _list
    return _list()


__all__ = [
    "Registry",
    "build_from_cfg",
    # Primitives
    "NamedRegistry",
    "VariantRegistry",
    # Training-level registries
    "MODELS_REGISTRY",
    "LOSSES_REGISTRY",
    "SAMPLERS_REGISTRY",
    "OPTIMIZERS_REGISTRY",
    "CONFIGS_REGISTRY",
    "LOGGERS_REGISTRY",
    "TRAINERS_REGISTRY",
    "DATASETS_REGISTRY",
    "METRICS_REGISTRY",
    # NN-component registries
    "BLOCKS_REGISTRY",
    "NORMS_REGISTRY",
    "COLLATORS_REGISTRY",
    "DATALOADERS_REGISTRY",
    "SCHEDULERS_REGISTRY",
    # Dict aliases
    "MODELS",
    "VARIANTS",
    "LOSSES",
    "SAMPLERS",
    "OPTIMIZERS",
    "CONFIGS",
    "LOGGERS",
    "TRAINERS",
    "DATASETS",
    "METRICS",
    "BLOCKS",
    "NORMS",
    "COLLATORS",
    "DATALOADERS",
    "SCHEDULERS",
    # Decorators
    "register_model",
    "register_variant",
    "register_loss",
    "register_sampler",
    "register_optimizer",
    "register_config",
    "register_logger",
    "register_trainer",
    "register_dataset",
    "register_metric",
    "register_block",
    "register_norm",
    "register_collator",
    "register_dataloader",
    "register_scheduler",
    # Getters
    "get_model",
    "get_variant",
    "get_loss",
    "get_sampler",
    "get_optimizer",
    "get_config_schema",
    "get_trainer",
    "get_dataset",
    "get_block",
    "get_norm",
    "get_collator",
    "get_dataloader_factory",
    "get_scheduler_factory",
    # Builders
    "build_logger",
    "build_norm",
    "build_attention_from_cfg",
    "build_ffn_from_cfg",
    "build_norm_from_cfg",
    "build_block_from_cfg",
    "build_backbone_from_cfg",
    "build_scheduler",
    "build_dataloader",
    # Lazy nn-registry accessors
    "get_attention_registry",
    "get_ffn_registry",
    "get_backbone_registry",
    # Vision backbone public surface
    "build_vision_backbone",
    "list_vision_backbones",
    "list_vision_backbones_by_family",
    # Misc
    "list_datasets",
    "instantiate_metrics",
]
