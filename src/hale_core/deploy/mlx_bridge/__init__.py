"""PyTorch → MLX graph bridge (``torch.export`` graph interpreted with MLX operators).

Importing this package does not import ``mlx``; :class:`AtenToMLXConverter` does on construction.
"""

from hale_core.deploy.mlx_bridge.aten_to_mlx_converter import PRESERVED_OPS, AtenToMLXConverter
from hale_core.deploy.mlx_bridge.mlx_node import MLXNode
from hale_core.deploy.mlx_bridge.mlx_program import MLXProgram
from hale_core.deploy.mlx_bridge.param_ref import ParamRef
from hale_core.deploy.mlx_bridge.value_ref import ValueRef

__all__ = ["PRESERVED_OPS", "AtenToMLXConverter", "MLXNode", "MLXProgram", "ParamRef", "ValueRef"]
