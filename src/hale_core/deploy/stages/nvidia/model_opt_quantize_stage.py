"""Post-training quantization of ONNX with NVIDIA ModelOpt (explicit Q/DQ for TensorRT)."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import Literal

import numpy as np

from hale_core.deploy.core.artifact import Artifact
from hale_core.deploy.core.command_runner import CommandRunner
from hale_core.deploy.core.export_context import ExportContext
from hale_core.deploy.core.optional_dependency import OptionalDependency
from hale_core.deploy.core.stage import Stage
from hale_core.deploy.registry import STAGES

MODELOPT = OptionalDependency("modelopt", "pip install nvidia-modelopt[onnx]")


@STAGES.register_module()
class ModelOptQuantizeStage(Stage):
    """Insert QuantizeLinear/DequantizeLinear pairs calibrated on the context's calibration data.

    TensorRT 11 has no implicit INT8 calibrator any more: quantization is expressed in the graph
    and the strongly typed engine follows it. Calibration batches are concatenated into one
    ``.npz`` (ModelOpt's ``--calibration_data`` format).

    Args:
        quantize_mode: ``int8`` (CNNs, Ampere+), ``fp8`` (Hopper / Ada / Blackwell),
            ``int4`` (weight-only, LLM linears).
        calibration_method: ``entropy`` / ``max`` (int8, fp8) or ``awq_clip`` / ``rtn_dq`` (int4).
        high_precision_dtype: dtype of the non-quantized parts (``fp16`` / ``bf16`` / ``fp32``).
        extra_args: any further ModelOpt CLI flags.
    """

    requires = ("onnx",)
    provides = ("onnx",)
    dependencies = (MODELOPT,)

    def __init__(
        self,
        quantize_mode: Literal["int8", "fp8", "int4"] = "int8",
        calibration_method: str | None = None,
        high_precision_dtype: str = "fp16",
        extra_args: Sequence[str] = (),
        **base,
    ) -> None:
        super().__init__(**base)
        self.quantize_mode = quantize_mode
        self.calibration_method = (
            calibration_method
            or {"int8": "entropy", "fp8": "max", "int4": "awq_clip"}[quantize_mode]
        )
        self.high_precision_dtype = high_precision_dtype
        self.extra_args = list(extra_args)

    def run(self, ctx: ExportContext) -> list[Artifact]:
        """Execute this stage on the pipeline context (see the class docstring).

        Args:
            ctx: the export context (model, signature, artifacts, work dir).

        Returns:
            The artifacts produced, also recorded in ``ctx.artifacts``.
        """
        source = ctx.store.require("onnx")
        stage_dir = ctx.stage_dir(self.name)
        calib = stage_dir / "calibration.npz"
        np.savez(calib, **ctx.calibration.stacked(ctx.signature))
        out = stage_dir / f"model.{self.quantize_mode}.onnx"
        cmd = [
            sys.executable,
            "-m",
            "modelopt.onnx.quantization",
            "--onnx_path",
            str(source.path),
            "--quantize_mode",
            self.quantize_mode,
            "--calibration_method",
            self.calibration_method,
            "--calibration_data",
            str(calib),
            "--high_precision_dtype",
            self.high_precision_dtype,
            "--output_path",
            str(out),
            *self.extra_args,
        ]
        CommandRunner().run(cmd)
        return [
            self.artifact(
                ctx,
                "onnx",
                out,
                quantization=self.quantize_mode,
                strongly_typed_ready=True,
                **{k: v for k, v in source.metadata.items() if k in ("opset", "inputs", "outputs")},
            )
        ]
