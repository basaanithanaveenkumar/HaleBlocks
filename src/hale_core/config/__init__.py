"""Typed run configuration, YAML loading with ``_base_`` inheritance and CLI overrides."""

from hale_core.config.config import Config
from hale_core.config.config_loader import ConfigLoader
from hale_core.config.dict_merger import DictMerger
from hale_core.config.experiment import apply_experiment_layout
from hale_core.config.override_parser import OverrideParser
from hale_core.config.run import RunConfig
from hale_core.config.sections import (
    DataConfig,
    EvalConfig,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    SampleConfig,
    TrainConfig,
    VizConfig,
)
from hale_core.registry.plugins import CONFIGS, register_config

register_config("llm")(RunConfig)

from hale_core.config.load import load_config, load_registered_config

__all__ = [
    "Config",
    "ConfigLoader",
    "DictMerger",
    "OverrideParser",
    "RunConfig",
    "load_config",
    "load_registered_config",
    "apply_experiment_layout",
    "ModelConfig",
    "TrainConfig",
    "DataConfig",
    "SampleConfig",
    "EvalConfig",
    "VizConfig",
    "LoggingConfig",
    "ExperimentConfig",
    "CONFIGS",
]
