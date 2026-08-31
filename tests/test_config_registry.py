import pytest
import yaml

from hale_core.config import RunConfig, load_registered_config
from hale_core.registry import (
    CONFIGS,
    LOGGERS,
    TRAINERS,
    build_logger,
    get_config_schema,
    get_trainer,
    register_config,
    register_logger,
    register_trainer,
)


def test_llm_config_registered():
    assert "llm" in CONFIGS
    assert get_config_schema("llm") is RunConfig


def test_register_custom_config_schema():
    @register_config("tiny")
    class TinyConfig(RunConfig):
        pass

    assert get_config_schema("tiny") is TinyConfig


def test_load_registered_config_from_yaml(tmp_path):
    path = tmp_path / "cfg.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "variant": "autoregressive",
                "model": {"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64, "max_length": 8},
                "train": {"batch_size": 2, "steps": 1},
                "logging": {"tensorboard": False, "log_file": None},
            }
        )
    )
    cfg = load_registered_config(path, schema="llm")
    assert isinstance(cfg, RunConfig)
    assert cfg.variant == "autoregressive"
    assert cfg.model.d_model == 32


def test_logger_registry_has_builtins():
    assert {"experiment", "noop", "loguru"} <= set(LOGGERS)


def test_build_noop_logger():
    cfg = RunConfig(
        variant="autoregressive",
        logging={"tensorboard": False, "log_file": None},
    )
    logger = build_logger("noop", cfg=cfg)
    logger.log_scalars(0, {"x": 1.0})
    logger.close()


def test_trainer_registry_has_default():
    assert "default" in TRAINERS
    assert get_trainer("default").__name__ == "Trainer"


def test_register_custom_trainer():
    @register_trainer("custom")
    class CustomTrainer:
        pass

    assert get_trainer("custom") is CustomTrainer
