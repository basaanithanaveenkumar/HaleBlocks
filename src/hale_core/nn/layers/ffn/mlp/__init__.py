"""hale_core.nn.layers.ffn.mlp."""

from hale_core.nn.layers.ffn.mlp.dense_mlp import DenseMLP
from hale_core.nn.layers.ffn.mlp.ge_glu import GeGLU

__all__ = ["DenseMLP", "GeGLU"]
