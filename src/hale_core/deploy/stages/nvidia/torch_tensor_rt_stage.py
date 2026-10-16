"""Compile with Torch-TensorRT (dynamo IR): TensorRT engines embedded in a PyTorch program.

Use it when a model has ops the ONNX parser rejects - unsupported subgraphs keep running in
PyTorch while the rest is lowered to TensorRT. The result is a serialized ``ExportedProgram``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from hale_core.deploy.core.artifact import Artifact
from hale_core.deploy.core.export_context import ExportContext
from hale_core.deploy.core.optional_dependency import OptionalDependency
from hale_core.deploy.core.stage import Stage
from hale_core.deploy.exceptions import MissingDependencyError
from hale_core.deploy.registry import STAGES

TORCH_TENSORRT = OptionalDependency(
    "torch_tensorrt", "pip install torch-tensorrt (CUDA build of torch)"
)


@STAGES.register_module()
class TorchTensorRTStage(Stage):
    """``torch_tensorrt.compile(ir="dynamo")`` with min/opt/max ``Input`` specs, then ``save``.

    Args:
        enabled_precisions: e.g. ``["float16"]`` or ``["float16", "float32"]``.
        min_block_size: smallest subgraph worth converting to TensorRT.
        workspace_gb: TensorRT workspace.
        filename: output file name.
    """

    provides = ("torch_tensorrt",)
    dependencies = (TORCH_TENSORRT,)

    def __init__(
        self,
        enabled_precisions: Sequence[str] = ("float16",),
        min_block_size: int = 3,
        workspace_gb: float = 4.0,
        filename: str = "model_trt.ep",
        **base,
    ) -> None:
        super().__init__(**base)
        self.enabled_precisions = list(enabled_precisions)
        self.min_block_size = min_block_size
        self.workspace_gb = workspace_gb
        self.filename = filename

    def run(self, ctx: ExportContext) -> list[Artifact]:
        if not torch.cuda.is_available():
            raise MissingDependencyError("cuda", "Torch-TensorRT needs an NVIDIA GPU")
        torch_trt = TORCH_TENSORRT.load()
        sig = ctx.signature
        inputs = [
            torch_trt.Input(
                min_shape=sig.shape(s, "min"),
                opt_shape=sig.shape(s, "opt"),
                max_shape=sig.shape(s, "max"),
                dtype=s.torch_dtype,
            )
            for s in sig.inputs
        ]
        module = ctx.export_module.cuda()
        compiled = torch_trt.compile(
            module,
            ir="dynamo",
            inputs=inputs,
            enabled_precisions={getattr(torch, p) for p in self.enabled_precisions},
            min_block_size=self.min_block_size,
            workspace_size=int(self.workspace_gb * (1 << 30)),
        )
        out = ctx.stage_dir(self.name) / self.filename
        args = tuple(t.cuda() for t in ctx.example_args("trace"))
        torch_trt.save(compiled, str(out), output_format="exported_program", arg_inputs=list(args))
        return [self.artifact(ctx, "torch_tensorrt", out, precisions=self.enabled_precisions)]
