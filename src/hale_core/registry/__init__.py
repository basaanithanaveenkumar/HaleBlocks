"""Decorator-based plugin registries.

Import submodules for side effects when adding new plugins:
  - ``hale_llm.models`` registers variants (model/loss/sampler/collate)
  - ``hale_core.nn.optim`` registers optimizers
  - ``hale_llm.metrics.llm`` registers generic metrics

Two registry primitives:
  - ``NamedRegistry``: one name -> one object (models, losses, samplers, optimizers, collate, ...)
  - ``VariantRegistry``: one name -> many objects, resolved by training variant (metrics)
"""

from hale_core.registry.base import NamedRegistry
from hale_core.registry.plugins import (
    CONFIGS,
    CONFIGS_REGISTRY,
    LOGGERS,
    LOGGERS_REGISTRY,
    LOSSES,
    LOSSES_REGISTRY,
    METRICS,
    METRICS_REGISTRY,
    MODELS,
    MODELS_REGISTRY,
    OPTIMIZERS,
    OPTIMIZERS_REGISTRY,
    SAMPLERS,
    SAMPLERS_REGISTRY,
    TRAINERS,
    TRAINERS_REGISTRY,
    DATASETS,
    DATASETS_REGISTRY,
    VARIANTS,
    build_logger,
    get_config_schema,
    get_loss,
    get_model,
    get_optimizer,
    get_sampler,
    get_trainer,
    get_dataset,
    list_datasets,
    get_variant,
    instantiate_metrics,
    register_config,
    register_logger,
    register_loss,
    register_metric,
    register_model,
    register_optimizer,
    register_sampler,
    register_trainer,
    register_dataset,
    register_variant,
)
from hale_core.registry.variant import VariantRegistry
from hale_core.registry.bootstrap import register_builtins

register_builtins()

__all__ = [
    "NamedRegistry",
    "VariantRegistry",
    "MODELS_REGISTRY",
    "LOSSES_REGISTRY",
    "SAMPLERS_REGISTRY",
    "OPTIMIZERS_REGISTRY",
    "CONFIGS_REGISTRY",
    "LOGGERS_REGISTRY",
    "TRAINERS_REGISTRY",
    "DATASETS_REGISTRY",
    "METRICS_REGISTRY",
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
    "get_model",
    "get_variant",
    "get_loss",
    "get_sampler",
    "get_optimizer",
    "get_config_schema",
    "get_trainer",
    "get_dataset",
    "list_datasets",
    "build_logger",
    "instantiate_metrics",
]
