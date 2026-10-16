"""Mixed-precision ONNX with NVIDIA ModelOpt AutoCast (the TensorRT 11 way to get FP16/BF16).

TensorRT 11 removed the ``kFP16`` / ``kBF16`` builder flags: engines are strongly typed and take
their precision from the ONNX graph. AutoCast rewrites an FP32 graph to FP16/BF16 where it is
numerically safe (keeping sensitive nodes such as large reductions in FP32).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Literal

from hale_core.deploy.core.artifact import Artifact
from hale_core.deploy.core.command_runner import CommandRunner
from hale_core.deploy.core.export_context import ExportContext
from hale_core.deploy.core.optional_dependency import OptionalDependency
from hale_core.deploy.core.stage import Stage
from hale_core.deploy.registry import STAGES

MODELOPT = OptionalDependency("modelopt", "pip install nvidia-modelopt[onnx]")


@STAGES.register_module()
class ModelOptAutoCastStage(Stage):
    """``python -m modelopt.onnx.autocast`` (CLI: stable across ModelOpt releases).

    Args:
        low_precision_type: ``fp16`` or ``bf16``.
        keep_io_types: keep FP32 graph inputs/outputs (callers keep feeding FP32).
        nodes_to_exclude: regexes of node names forced to stay FP32.
        extra_args: any further AutoCast CLI flags.
    """

    requires = ("onnx",)
    provides = ("onnx",)
    dependencies = (MODELOPT,)

    def __init__(
        self,
        low_precision_type: Literal["fp16", "bf16"] = "fp16",
        keep_io_types: bool = True,
        nodes_to_exclude: Sequence[str] = (),
        extra_args: Sequence[str] = (),
        **base,
    ) -> None:
        super().__init__(**base)
        self.low_precision_type = low_precision_type
        self.keep_io_types = keep_io_types
        self.nodes_to_exclude = list(nodes_to_exclude)
        self.extra_args = list(extra_args)

    def command(self, source: str, output: str) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "modelopt.onnx.autocast",
            "--onnx_path",
            source,
            "--output_path",
            output,
            "--low_precision_type",
            self.low_precision_type,
        ]
        if self.keep_io_types:
            cmd.append("--keep_io_types")
        if self.nodes_to_exclude:
            cmd += ["--nodes_to_exclude", *self.nodes_to_exclude]
        return cmd + self.extra_args

    def run(self, ctx: ExportContext) -> list[Artifact]:
        source = ctx.store.require("onnx")
        out = ctx.stage_dir(self.name) / f"model.{self.low_precision_type}.onnx"
        CommandRunner().run(self.command(str(source.path), str(out)))
        return [
            self.artifact(
                ctx,
                "onnx",
                out,
                precision=self.low_precision_type,
                strongly_typed_ready=True,
                **{k: v for k, v in source.metadata.items() if k in ("opset", "inputs", "outputs")},
            )
        ]
