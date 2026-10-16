"""Execute a serialized TensorRT engine (TensorRT 10 and 11 name-based tensor API)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

from hale_core.deploy.core.optional_dependency import OptionalDependency
from hale_core.deploy.registry import RUNTIMES
from hale_core.deploy.runtimes.inference_session import InferenceSession

TENSORRT = OptionalDependency(
    "tensorrt", "install TensorRT (pip install tensorrt) on a CUDA machine"
)


@RUNTIMES.register_module()
class TensorRTSession(InferenceSession):
    """Deserialize an engine and run it with ``execute_async_v3`` on CUDA tensors.

    Device buffers are torch CUDA tensors (no pycuda dependency); dynamic input shapes are set
    per call with ``set_input_shape`` and output shapes are read back from the context.

    Args:
        path: ``.engine`` file.
        runtime_module: ``tensorrt`` or ``tensorrt_rtx`` (TensorRT for RTX, same tensor API).
    """

    dependencies = (TENSORRT,)

    def __init__(self, path: str | Path, runtime_module: str = "tensorrt") -> None:
        super().__init__(path)
        import torch

        trt = OptionalDependency(runtime_module).load()
        if not torch.cuda.is_available():
            from hale_core.deploy.exceptions import MissingDependencyError

            raise MissingDependencyError("cuda", "TensorRT engines run on NVIDIA GPUs only")
        self.trt = trt
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.runtime.deserialize_cuda_engine(self.path.read_bytes())
        self.context = self.engine.create_execution_context()
        names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
        self.inputs = [n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [
            n for n in names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT
        ]

    def run(self, inputs: Mapping[str, np.ndarray]) -> list[np.ndarray]:
        import torch

        stream = torch.cuda.Stream()
        keep = []
        for name in self.inputs:
            tensor = torch.from_numpy(np.ascontiguousarray(inputs[name])).cuda()
            keep.append(tensor)
            self.context.set_input_shape(name, tuple(tensor.shape))
            self.context.set_tensor_address(name, tensor.data_ptr())
        outs = []
        for name in self.outputs:
            dtype = torch.from_numpy(
                np.empty(0, dtype=self.trt.nptype(self.engine.get_tensor_dtype(name)))
            ).dtype
            out = torch.empty(
                tuple(self.context.get_tensor_shape(name)), dtype=dtype, device="cuda"
            )
            self.context.set_tensor_address(name, out.data_ptr())
            outs.append(out)
        if not self.context.execute_async_v3(stream.cuda_stream):
            from hale_core.deploy.exceptions import DeployError

            raise DeployError(f"TensorRT execution failed for {self.path}")
        stream.synchronize()
        return [o.cpu().numpy() for o in outs]
