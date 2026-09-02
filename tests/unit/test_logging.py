import pytest

from hale_core.config import RunConfig
from hale_core.registry import build_logger


def test_build_noop_logger():
    cfg = RunConfig(
        variant="mock_ar",
        logging={"tensorboard": False, "log_file": None, "backend": "noop"},
    )
    logger = build_logger("noop", cfg=cfg)
    logger.log_scalars(0, {"x": 1.0})
    logger.close()


def test_build_unknown_logger_raises():
    cfg = RunConfig(variant="mock_ar", logging={"backend": "noop"})
    with pytest.raises(KeyError):
        build_logger("does_not_exist", cfg=cfg)
