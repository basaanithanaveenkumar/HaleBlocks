"""Compile ONNX into a MIGraphX program (``.mxr``) for AMD Instinct / Radeon GPUs (ROCm).

ROCm 7.1+ removed ONNX Runtime's ROCm EP; MIGraphX (directly, or as ORT's
``MIGraphXExecutionProvider``) is AMD's supported inference path. Compiling ahead of time
avoids the multi-minute JIT on first request in production.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from loguru import logger

from hale_core.deploy.core.artifact import Artifact
from hale_core.deploy.core.command_runner import CommandRunner
from hale_core.deploy.core.export_context import ExportContext
from hale_core.deploy.core.optional_dependency import OptionalDependency
from hale_core.deploy.core.stage import Stage
from hale_core.deploy.exceptions import MissingDependencyError
from hale_core.deploy.registry import STAGES

MIGRAPHX = OptionalDependency(
    "migraphx", "install ROCm with MIGraphX (python3-migraphx); or put migraphx-driver on PATH"
)


@STAGES.register_module()
class MIGraphXCompileStage(Stage):
    """Python API (``parse_onnx`` → quantize → ``compile`` → ``save``), CLI fallback.

    Args:
        precision: ``fp32``, ``fp16``, ``bf16`` or ``int8`` (int8 calibrates on the context's
            calibration batches).
        target: MIGraphX target (``gpu``; ``cpu`` / ``ref`` for testing).
        offload_copy: let MIGraphX copy inputs/outputs host↔device (simplest serving setup).
        exhaustive_tune: benchmark kernels during compile (slower compile, faster runtime).
        shape: signature corner whose static shapes the program is compiled for.
    """

    requires = ("onnx",)
    provides = ("migraphx_program",)

    def __init__(
        self,
        precision: Literal["fp32", "fp16", "bf16", "int8"] = "fp16",
        target: str = "gpu",
        offload_copy: bool = True,
        exhaustive_tune: bool = False,
        shape: str = "opt",
        filename: str = "model.mxr",
        **base,
    ) -> None:
        super().__init__(**base)
        self.precision = precision
        self.target = target
        self.offload_copy = offload_copy
        self.exhaustive_tune = exhaustive_tune
        self.shape = shape
        self.filename = filename

    def missing_dependencies(self):
        """Optional packages or executables this stage needs that are not installed."""
        if MIGRAPHX.available:
            return []
        try:
            CommandRunner.which("migraphx-driver")
            return []
        except MissingDependencyError:
            return [MIGRAPHX]

    def run(self, ctx: ExportContext) -> list[Artifact]:
        """Execute this stage on the pipeline context (see the class docstring).

        Args:
            ctx: the export context (model, signature, artifacts, work dir).

        Returns:
            The artifacts produced, also recorded in ``ctx.artifacts``.
        """
        source = ctx.store.require("onnx")
        out = ctx.stage_dir(self.name) / self.filename
        dims = {s.name: list(ctx.signature.shape(s, self.shape)) for s in ctx.signature.inputs}
        if MIGRAPHX.available:
            self._compile_python(ctx, str(source.path), str(out), dims)
        else:
            self._compile_cli(str(source.path), str(out), dims)
        return [
            self.artifact(
                ctx,
                "migraphx_program",
                out,
                precision=self.precision,
                target=self.target,
                shapes=dims,
            )
        ]

    def _compile_python(
        self, ctx: ExportContext, source: str, out: str, dims: dict[str, list[int]]
    ) -> None:
        migraphx = MIGRAPHX.load()
        program = migraphx.parse_onnx(source, map_input_dims=dims)
        target = migraphx.get_target(self.target)
        if self.precision == "fp16":
            migraphx.quantize_fp16(program)
        elif self.precision == "bf16":
            migraphx.quantize_bf16(program)
        elif self.precision == "int8":
            calibration = [
                {k: migraphx.argument(np.ascontiguousarray(v)) for k, v in batch.items()}
                for batch in ctx.calibration.as_numpy(ctx.signature)
            ]
            migraphx.quantize_int8(program, target, calibration)
        program.compile(
            target, offload_copy=self.offload_copy, exhaustive_tune=self.exhaustive_tune
        )
        migraphx.save(program, out, format="msgpack")
        logger.info("MIGraphX program saved {}", out)

    def _compile_cli(self, source: str, out: str, dims: dict[str, list[int]]) -> None:
        if self.precision in ("bf16", "int8"):
            logger.warning(
                "migraphx-driver CLI path supports fp16/fp32 here; compiling {} as fp16",
                self.precision,
            )
        cmd = [
            CommandRunner.which("migraphx-driver"),
            "compile",
            "--onnx",
            source,
            f"--{self.target}",
            "--binary",
            "-o",
            out,
        ]
        for name, shape in dims.items():
            cmd += ["--input-dim", f"@{name}", *map(str, shape)]
        if self.precision != "fp32":
            cmd.append("--fp16")
        if self.exhaustive_tune:
            cmd.append("--exhaustive-tune")
        CommandRunner().run(cmd)
