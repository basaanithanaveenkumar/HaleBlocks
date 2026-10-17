"""Run an exported MLX function (``.mlxfn``) on Apple Silicon."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from hale_core.deploy.core.optional_dependency import OptionalDependency
from hale_core.deploy.registry import RUNTIMES
from hale_core.deploy.runtimes.inference_session import InferenceSession

MLX = OptionalDependency("mlx", "pip install mlx (macOS on Apple Silicon)")


@RUNTIMES.register_module()
class MLXFunctionSession(InferenceSession):
    """``mx.import_function`` + positional call in signature order.

    Args:
        path: ``.mlxfn`` written by :class:`MLXConvertStage`.
        input_names: positional order of the inputs.
    """

    dependencies = (MLX,)

    def __init__(self, path: str | Path, input_names: Sequence[str]) -> None:
        super().__init__(path)
        import mlx.core as mx

        self.mx = mx
        self.fn = mx.import_function(str(self.path))
        self.input_names = list(input_names)

    def run(self, inputs: Mapping[str, np.ndarray]) -> list[np.ndarray]:
        """Run one inference.

        Args:
            inputs: input name -> array, matching the exported signature.

        Returns:
            Output arrays in signature order.
        """
        outputs = self.fn(*(self.mx.array(inputs[n]) for n in self.input_names))
        return [
            np.array(o.astype(self.mx.float32) if o.dtype in (self.mx.bfloat16,) else o)
            for o in outputs
        ]
