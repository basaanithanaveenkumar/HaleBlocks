from pathlib import Path

import pytest
from pydantic import ValidationError

from hale_core.config import RunConfig, load_config
from hale_core.training.schedule import resolve_schedule

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def test_bad_key_fails():
    with pytest.raises(ValidationError):
        RunConfig.model_validate({"variant": "mock_ar", "model": {"not_a_field": 1}})


def test_resolve_schedule_epochs_wins():
    cfg = RunConfig(variant="mock_ar", train={"epochs": 3, "steps": 999})
    n_epochs, total_steps = resolve_schedule(cfg, n_batches=10)
    assert n_epochs == 3
    assert total_steps == 30


def test_resolve_schedule_steps_only():
    cfg = RunConfig(variant="mock_ar", train={"epochs": None, "steps": 25})
    n_epochs, total_steps = resolve_schedule(cfg, n_batches=10)
    assert total_steps == 25
    assert n_epochs == 3


def test_yaml_load_minimal():
    cfg = load_config(FIXTURES / "minimal.yaml")
    assert cfg.variant == "mock_ar"
    assert cfg.model.d_model == 32
    assert cfg.train.steps == 4
    assert cfg.logging.backend == "noop"


def test_yaml_inherit_merge():
    cfg = load_config(FIXTURES / "child.yaml")
    assert cfg.variant == "mock_ar"
    assert cfg.model.d_model == 48
    assert cfg.train.steps == 8
