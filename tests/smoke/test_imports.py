import pytest


@pytest.mark.smoke
def test_import_hale_core():
    import hale_core

    assert hale_core.__version__ == "0.1.0"


@pytest.mark.smoke
def test_import_subpackages():
    import hale_core.config
    import hale_core.logging
    import hale_core.nn
    import hale_core.registry
    import hale_core.runtime
    import hale_core.training


@pytest.mark.smoke
def test_registry_builtins_loaded():
    from hale_core.registry import CONFIGS, LOGGERS, OPTIMIZERS, TRAINERS

    assert "llm" in CONFIGS
    assert "experiment" in LOGGERS
    assert "adamw" in OPTIMIZERS
    assert "default" in TRAINERS
