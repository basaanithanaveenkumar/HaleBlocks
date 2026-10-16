"""mmdet-style registries of the deployment framework."""

from hale_core.deploy.registry.registries import ADAPTERS, CALIBRATION, MLX_OPS, RUNTIMES, STAGES
from hale_core.deploy.registry.registry import Registry, build_from_cfg

__all__ = ["ADAPTERS", "CALIBRATION", "MLX_OPS", "RUNTIMES", "STAGES", "Registry", "build_from_cfg"]
