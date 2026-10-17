"""Export the adapter's model to ONNX - the interchange format for TensorRT, MIGraphX, Quark,
ONNX Runtime EPs (QNN, Vitis AI, MIGraphX, TensorRT) and Qualcomm AI Hub."""

from __future__ import annotations

from typing import Literal

import torch
from loguru import logger

from hale_core.deploy.core.artifact import Artifact
from hale_core.deploy.core.export_context import ExportContext
from hale_core.deploy.core.optional_dependency import OptionalDependency
from hale_core.deploy.core.stage import Stage
from hale_core.deploy.exceptions import StageError
from hale_core.deploy.registry import STAGES

ONNX = OptionalDependency("onnx", "pip install onnx")
ONNXSCRIPT = OptionalDependency(
    "onnxscript", "pip install onnxscript (needed by the dynamo exporter)"
)


@STAGES.register_module()
class OnnxExportStage(Stage):
    """``torch.onnx.export`` with signature-derived names and dynamic dimensions.

    Args:
        opset: ONNX opset. 18+ is required by the dynamo exporter; TensorRT 10/11, MIGraphX
            and ORT 1.2x all accept 18-21.
        exporter: ``dynamo`` (``torch.export`` based, default since torch 2.9), ``torchscript``
            (legacy tracer) or ``auto`` (dynamo, falling back to torchscript on failure).
        external_data: store weights next to the graph (``model.onnx.data``); needed above the
            2 GB protobuf limit. ``None`` decides from the parameter size.
        optimize: run the dynamo exporter's graph optimizer (constant folding, fusion).
        filename: output file name.
    """

    provides = ("onnx",)
    dependencies = (ONNX,)

    def __init__(
        self,
        opset: int = 18,
        exporter: Literal["dynamo", "torchscript", "auto"] = "auto",
        external_data: bool | None = None,
        optimize: bool = True,
        filename: str = "model.onnx",
        **base,
    ) -> None:
        super().__init__(**base)
        self.opset = opset
        self.exporter = exporter
        self.external_data = external_data
        self.optimize = optimize
        self.filename = filename

    def run(self, ctx: ExportContext) -> list[Artifact]:
        """Execute this stage on the pipeline context (see the class docstring).

        Args:
            ctx: the export context (model, signature, artifacts, work dir).

        Returns:
            The artifacts produced, also recorded in ``ctx.artifacts``.
        """
        path = ctx.stage_dir(self.name) / self.filename
        sig = ctx.signature
        module, args = ctx.export_module, ctx.example_args("trace")
        params = sum(p.numel() * p.element_size() for p in module.parameters())
        external = self.external_data if self.external_data is not None else params > 1.8e9
        order = {
            "dynamo": ["dynamo"],
            "torchscript": ["torchscript"],
            "auto": ["dynamo", "torchscript"],
        }[self.exporter]
        errors = []
        for exporter in order:
            try:
                self._export(exporter, module, args, sig, path, external)
                break
            except Exception as exc:  # noqa: BLE001 - exporter failures are diverse; try the next one
                logger.warning("{} ONNX exporter failed: {}", exporter, exc)
                errors.append(f"{exporter}: {type(exc).__name__}: {exc}")
        else:
            raise StageError(self.name, "all ONNX exporters failed:\n" + "\n".join(errors))
        fallback = exporter != order[0]
        if fallback:
            logger.warning(
                "ONNX export fell back from {} to {}: dynamic shapes come from "
                "dynamic_axes, not torch.export constraints",
                order[0],
                exporter,
            )
        onnx = ONNX.load()
        onnx.checker.check_model(str(path))
        graph = onnx.load(str(path), load_external_data=False).graph
        logger.info(
            "ONNX {} ({} nodes, opset {}, exporter {})", path, len(graph.node), self.opset, exporter
        )
        return [
            self.artifact(
                ctx,
                "onnx",
                path,
                opset=self.opset,
                exporter=exporter,
                fallback=fallback,
                external_data=external,
                inputs=sig.input_names,
                outputs=list(sig.outputs),
            )
        ]

    def _export(self, exporter: str, module, args, sig, path, external: bool) -> None:
        with torch.no_grad():
            if exporter == "dynamo":
                ONNXSCRIPT.load()
                torch.onnx.export(
                    module,
                    args,
                    str(path),
                    dynamo=True,
                    opset_version=self.opset,
                    input_names=sig.input_names,
                    output_names=list(sig.outputs),
                    dynamic_shapes=sig.torch_dynamic_shapes(),
                    external_data=external,
                    optimize=self.optimize,
                )
            else:
                axes = sig.onnx_dynamic_axes()
                torch.onnx.export(
                    module,
                    args,
                    str(path),
                    dynamo=False,
                    opset_version=self.opset,
                    input_names=sig.input_names,
                    output_names=list(sig.outputs),
                    dynamic_axes=axes or None,
                    do_constant_folding=True,
                )
