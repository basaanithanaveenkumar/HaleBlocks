"""Run a compiled MIGraphX program (``.mxr``) on an AMD GPU."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from hale_core.deploy.core.optional_dependency import OptionalDependency
from hale_core.deploy.registry import RUNTIMES
from hale_core.deploy.runtimes.inference_session import InferenceSession

MIGRAPHX = OptionalDependency(
    "migraphx", "install ROCm + MIGraphX (python3-migraphx) on an AMD GPU machine"
)


@RUNTIMES.register_module()
class MIGraphXSession(InferenceSession):
    """``migraphx.load`` + ``program.run`` with ``migraphx.argument`` inputs."""

    dependencies = (MIGRAPHX,)

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        self.migraphx = MIGRAPHX.load()
        self.program = self.migraphx.load(str(self.path), format="msgpack")

    def run(self, inputs: Mapping[str, np.ndarray]) -> list[np.ndarray]:
        args = {
            name: self.migraphx.argument(np.ascontiguousarray(value))
            for name, value in inputs.items()
        }
        return [np.array(result) for result in self.program.run(args)]
