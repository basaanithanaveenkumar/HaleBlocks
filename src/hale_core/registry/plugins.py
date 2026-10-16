"""Training/eval plugin registries.

NamedRegistry plugins (strict 1:1 name lookup):
  model, loss, sampler, optimizer, config, logger, trainer, dataset,
  block, norm, collator, dataloader, scheduler

VariantRegistry plugins (name + variant resolution):
  metric
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from hale_core.registry.base import NamedRegistry
from hale_core.registry.variant import VariantRegistry

T = TypeVar("T")

# ── Training-level ────────────────────────────────────────────────────────────
MODELS_REGISTRY = NamedRegistry("model")
LOSSES_REGISTRY = NamedRegistry("loss")
SAMPLERS_REGISTRY = NamedRegistry("sampler")
OPTIMIZERS_REGISTRY = NamedRegistry("optimizer")
CONFIGS_REGISTRY = NamedRegistry("config")
LOGGERS_REGISTRY = NamedRegistry("logger")
TRAINERS_REGISTRY = NamedRegistry("trainer")
DATASETS_REGISTRY = NamedRegistry("dataset")
METRICS_REGISTRY = VariantRegistry("metric")

# ── NN-component level ────────────────────────────────────────────────────────
BLOCKS_REGISTRY = NamedRegistry("block")
NORMS_REGISTRY = NamedRegistry("norm")
COLLATORS_REGISTRY = NamedRegistry("collator")
DATALOADERS_REGISTRY = NamedRegistry("dataloader")
SCHEDULERS_REGISTRY = NamedRegistry("scheduler")

# ── Convenience aliases (dict views) ─────────────────────────────────────────
MODELS: dict[str, type] = MODELS_REGISTRY.items
VARIANTS = MODELS
LOSSES: dict[str, Callable[..., Any]] = LOSSES_REGISTRY.items
SAMPLERS: dict[str, Callable[..., Any]] = SAMPLERS_REGISTRY.items
OPTIMIZERS: dict[str, type] = OPTIMIZERS_REGISTRY.items
CONFIGS: dict[str, type] = CONFIGS_REGISTRY.items
LOGGERS: dict[str, Callable[..., Any]] = LOGGERS_REGISTRY.items
TRAINERS: dict[str, type] = TRAINERS_REGISTRY.items
DATASETS: dict[str, Callable[..., Any]] = DATASETS_REGISTRY.items
METRICS = METRICS_REGISTRY.items
BLOCKS: dict[str, type] = BLOCKS_REGISTRY.items
NORMS: dict[str, type] = NORMS_REGISTRY.items
COLLATORS: dict[str, Callable[..., Any]] = COLLATORS_REGISTRY.items
DATALOADERS: dict[str, Callable[..., Any]] = DATALOADERS_REGISTRY.items
SCHEDULERS: dict[str, type] = SCHEDULERS_REGISTRY.items


def register_model(name: str) -> Callable[[type[T]], type[T]]:
    def deco(cls: type[T]) -> type[T]:
        MODELS_REGISTRY.add(name, cls)
        cls.variant_name = name  # type: ignore[attr-defined]
        return cls

    return deco


register_variant = register_model


def register_loss(name: str) -> Callable[[T], T]:
    def deco(fn: T) -> T:
        LOSSES_REGISTRY.add(name, fn)
        return fn

    return deco


def register_sampler(name: str) -> Callable[[T], T]:
    def deco(fn: T) -> T:
        SAMPLERS_REGISTRY.add(name, fn)
        return fn

    return deco


def register_optimizer(name: str) -> Callable[[type[T]], type[T]]:
    def deco(cls: type[T]) -> type[T]:
        OPTIMIZERS_REGISTRY.add(name.lower(), cls)
        return cls

    return deco


def register_metric(
    name: str, *, variants: tuple[str, ...] | None = None
) -> Callable[[type[T]], type[T]]:
    def deco(cls: type[T]) -> type[T]:
        return METRICS_REGISTRY.add(name, cls, variants=variants)

    return deco


def get_model(name: str) -> type:
    return MODELS_REGISTRY.get(name)


get_variant = get_model


def get_loss(name: str) -> Callable[..., Any]:
    return LOSSES_REGISTRY.get(name)


def get_sampler(name: str) -> Callable[..., Any]:
    return SAMPLERS_REGISTRY.get(name)


def get_optimizer(name: str) -> type:
    return OPTIMIZERS_REGISTRY.get(name.lower())


def register_config(name: str) -> Callable[[type[T]], type[T]]:
    def deco(cls: type[T]) -> type[T]:
        CONFIGS_REGISTRY.add(name, cls)
        return cls

    return deco


def get_config_schema(name: str = "llm") -> type:
    return CONFIGS_REGISTRY.get(name)


def register_logger(name: str) -> Callable[[T], T]:
    def deco(factory: T) -> T:
        LOGGERS_REGISTRY.add(name, factory)
        return factory

    return deco


def build_logger(name: str, *, cfg: Any = None, **kwargs: Any) -> Any:
    return LOGGERS_REGISTRY.get(name)(cfg=cfg, **kwargs)


def register_trainer(name: str) -> Callable[[type[T]], type[T]]:
    def deco(cls: type[T]) -> type[T]:
        TRAINERS_REGISTRY.add(name, cls)
        return cls

    return deco


def get_trainer(name: str = "default") -> type:
    return TRAINERS_REGISTRY.get(name)


def register_dataset(name: str) -> Callable[[T], T]:
    def deco(factory: T) -> T:
        DATASETS_REGISTRY.add(name, factory)
        return factory

    return deco


def get_dataset(name: str) -> Any:
    factory = DATASETS_REGISTRY.get(name)
    return factory()


def _spec_attr_matches(value: Any, expected: str) -> bool:
    if value is None:
        return False
    actual = value.value if hasattr(value, "value") else str(value)
    return actual == expected or str(value) == expected


def list_datasets(
    *,
    kind: str | None = None,
    stage: str | None = None,
    domain: str | None = None,
) -> list[str]:
    names = sorted(DATASETS_REGISTRY.items)
    if not kind and not stage and not domain:
        return names
    filtered: list[str] = []
    for name in names:
        spec = get_dataset(name)
        if kind is not None and not _spec_attr_matches(getattr(spec, "kind", None), kind):
            continue
        if stage is not None and not _spec_attr_matches(getattr(spec, "stage", None), stage):
            continue
        if domain is not None and not _spec_attr_matches(getattr(spec, "domain", None), domain):
            continue
        filtered.append(name)
    return filtered


def instantiate_metrics(names: list[str], variant: str) -> list[Any]:
    return METRICS_REGISTRY.instantiate(names, variant)


# ── Block ─────────────────────────────────────────────────────────────────────


def register_block(name: str) -> Callable[[type[T]], type[T]]:
    """Register a transformer block class under *name*.

    Example::

        @register_block("transformer")
        class TransformerLayer(nn.Module): ...
    """

    def deco(cls: type[T]) -> type[T]:
        BLOCKS_REGISTRY.add(name, cls)
        return cls

    return deco


def get_block(name: str) -> type:
    return BLOCKS_REGISTRY.get(name)


# ── Norm ──────────────────────────────────────────────────────────────────────


def register_norm(name: str) -> Callable[[type[T]], type[T]]:
    """Register a normalisation layer class under *name*.

    Example::

        @register_norm("rms")
        class RMSNorm(nn.Module): ...
    """

    def deco(cls: type[T]) -> type[T]:
        NORMS_REGISTRY.add(name, cls)
        return cls

    return deco


def get_norm(name: str) -> type:
    return NORMS_REGISTRY.get(name)


def build_norm(name: str, d_model: int, **kwargs: Any) -> Any:
    """Instantiate a registered norm by *name* for dimension *d_model*."""
    cls = NORMS_REGISTRY.get(name)
    return cls(d_model, **kwargs)


# ── Collator ──────────────────────────────────────────────────────────────────


def register_collator(name: str) -> Callable[[T], T]:
    """Register a collate function.

    Example::

        @register_collator("pad_sequence")
        def pad_collate(batch): ...
    """

    def deco(fn: T) -> T:
        COLLATORS_REGISTRY.add(name, fn)
        return fn

    return deco


def get_collator(name: str) -> Callable[..., Any]:
    return COLLATORS_REGISTRY.get(name)


# ── Dataloader ────────────────────────────────────────────────────────────────


def register_dataloader(name: str) -> Callable[[T], T]:
    """Register a DataLoader factory function.

    The factory signature is ``factory(dataset, cfg: DataLoaderConfig, batch_size: int)``.

    Example::

        @register_dataloader("default")
        def default_dataloader(dataset, cfg, batch_size): ...
    """

    def deco(fn: T) -> T:
        DATALOADERS_REGISTRY.add(name, fn)
        return fn

    return deco


def get_dataloader_factory(name: str) -> Callable[..., Any]:
    return DATALOADERS_REGISTRY.get(name)


# ── Scheduler ─────────────────────────────────────────────────────────────────


def register_scheduler(name: str) -> Callable[[type[T]], type[T]]:
    """Register an LR-scheduler class or factory under *name*.

    The factory signature is
    ``factory(optimizer, cfg: SchedulerConfig, total_steps: int)``.

    Example::

        @register_scheduler("cosine")
        def cosine_scheduler(optimizer, cfg, total_steps): ...
    """

    def deco(fn_or_cls: type[T]) -> type[T]:
        SCHEDULERS_REGISTRY.add(name, fn_or_cls)
        return fn_or_cls

    return deco


def get_scheduler_factory(name: str) -> Callable[..., Any]:
    return SCHEDULERS_REGISTRY.get(name)
