"""ATen → MLX op converters. Importing this package registers them in ``MLX_OPS``.

Requires ``mlx`` (Apple Silicon); only the MLX bridge imports it, lazily.
"""

from hale_core.deploy.mlx_bridge.ops import (  # noqa: F401
    creation_ops,
    elementwise_ops,
    linalg_ops,
    nn_ops,
    reduction_ops,
    shape_ops,
)

__all__ = ["creation_ops", "elementwise_ops", "linalg_ops", "nn_ops", "reduction_ops", "shape_ops"]
