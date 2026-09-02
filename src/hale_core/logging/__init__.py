from __future__ import annotations

from typing import Any

from hale_core.config.run import RunConfig
from hale_core.logging.experiment import ExperimentLogger, make_experiment_logger
from hale_core.logging.setup import setup_logging
from hale_core.registry import register_logger


class _NoOpExperimentLogger:
    def log_scalars(self, step: int, metrics: dict[str, float]) -> None:
        return None

    def log_hparams(self, hparams: dict[str, Any], metrics: dict[str, float] | None = None) -> None:
        return None

    def log_text(self, tag: str, text: str, step: int = 0) -> None:
        return None

    def log_image(self, tag: str, image, step: int = 0) -> None:
        return None

    def log_histogram(self, tag: str, values, step: int = 0) -> None:
        return None

    def watch_model(self, model, log_freq: int = 100) -> None:
        return None

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


@register_logger("noop")
def _noop_logger(*, cfg: RunConfig | None = None, **kwargs) -> _NoOpExperimentLogger:
    return _NoOpExperimentLogger()


@register_logger("experiment")
def _experiment_logger(*, cfg: RunConfig | None = None, **kwargs) -> ExperimentLogger:
    if cfg is None:
        raise TypeError("experiment logger requires cfg=RunConfig")
    run_name = None if cfg.experiment.enabled else cfg.variant
    if cfg.experiment.enabled and cfg.experiment.name:
        run_name = cfg.experiment.name
    main_process = kwargs.get("main_process", True)
    return make_experiment_logger(
        tensorboard_enabled=cfg.logging.tensorboard and main_process,
        tensorboard_dir=cfg.logging.tensorboard_dir,
        wandb_enabled=cfg.logging.wandb and main_process,
        wandb_project=cfg.logging.wandb_project,
        wandb_entity=cfg.logging.wandb_entity,
        wandb_run_name=cfg.logging.wandb_run_name,
        wandb_tags=cfg.logging.wandb_tags,
        run_name=run_name,
    )


@register_logger("loguru")
def _loguru_logger(*, cfg: RunConfig | None = None, **kwargs) -> None:
    if cfg is None:
        raise TypeError("loguru logger requires cfg=RunConfig")
    setup_logging(
        level=cfg.logging.level, log_file=cfg.logging.log_file, force=kwargs.get("force", True)
    )
    return None


__all__ = [
    "ExperimentLogger",
    "make_experiment_logger",
    "setup_logging",
]
