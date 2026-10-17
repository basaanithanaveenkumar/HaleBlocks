"""Load and inherit YAML into a validated config schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from hale_core.config.config_loader import ConfigLoader
from hale_core.config.run import RunConfig
from hale_core.registry.plugins import get_config_schema


def load_yaml_tree(path: str | Path) -> dict[str, Any]:
    """Load ``path`` with its whole ``inherits:`` / ``_base_`` chain merged in.

    Delegates to :class:`ConfigLoader`, so inheritance can be any depth, may list several
    bases, may use ``pkg://`` references and ``_delete_`` replacement, and cycles raise
    :class:`~hale_core.exceptions.ConfigError` instead of recursing forever.

    Raises:
        FileNotFoundError: if ``path`` does not exist (kept for existing callers).
        ConfigError: for unreadable files or cyclic inheritance.
    """
    path = Path(path).resolve()
    logger.debug("loading config from {}", path)
    if not path.exists():
        logger.error("config file not found: {}", path)
        raise FileNotFoundError(path)
    return ConfigLoader().load(path)


def validate_config[T: BaseModel](raw: dict[str, Any], schema: type[T]) -> T:
    """Validate a merged raw config into ``schema``, logging the payload on failure.

    Args:
        raw: merged config dict.
        schema: pydantic model class.

    Returns:
        The validated model instance.

    Raises:
        pydantic.ValidationError: if ``raw`` does not satisfy the schema.
    """
    try:
        return schema.model_validate(raw)
    except Exception:
        logger.exception("invalid config payload for schema {}", schema.__name__)
        raise


def load_registered_config(path: str | Path, schema: str = "llm") -> BaseModel:
    """Load ``path`` and validate it with the config schema registered as ``schema``.

    Args:
        path: YAML config (inheritance resolved by :func:`load_yaml_tree`).
        schema: key in the ``CONFIGS`` registry (``"llm"`` = :class:`RunConfig`; downstream
            packages register their own, e.g. ``"vlm"``).

    Returns:
        The validated config; ``experiment.source_yaml`` is set when the schema has it.
    """
    import hale_core.config  # noqa: F401 — ensure default schemas register

    path = Path(path).resolve()
    model_cls = get_config_schema(schema)
    cfg = validate_config(load_yaml_tree(path), model_cls)
    if hasattr(cfg, "experiment") and getattr(cfg.experiment, "source_yaml", None) is not None:
        cfg.experiment.source_yaml = str(path)
    logger.info(
        "loaded config schema={} variant={} from {}", schema, getattr(cfg, "variant", "?"), path
    )
    return cfg


def load_config(path: str | Path) -> RunConfig:
    """Load ``path`` as an LLM :class:`RunConfig` (the ``"llm"`` schema)."""
    cfg = load_registered_config(path, schema="llm")
    assert isinstance(cfg, RunConfig)
    return cfg
