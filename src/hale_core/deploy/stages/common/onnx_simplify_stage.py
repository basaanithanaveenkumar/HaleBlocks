"""Shape inference and (optionally) onnx-simplifier on the latest ONNX artifact."""

from __future__ import annotations

from loguru import logger

from hale_core.deploy.core.artifact import Artifact
from hale_core.deploy.core.export_context import ExportContext
from hale_core.deploy.core.optional_dependency import OptionalDependency
from hale_core.deploy.core.stage import Stage
from hale_core.deploy.registry import STAGES

ONNX = OptionalDependency("onnx", "pip install onnx")
ONNXSIM = OptionalDependency("onnxsim", "pip install onnxsim")


@STAGES.register_module()
class OnnxSimplifyStage(Stage):
    """Make the graph friendlier for vendor parsers.

    * ``onnx.shape_inference`` annotates every intermediate shape (MIGraphX and the QNN
      converter rely on it);
    * ``onnxsim`` (when installed and ``simplify``) folds constants and removes the shape
      arithmetic exporters leave behind - often the difference between a TensorRT parse
      error and a clean build. Skipped above 2 GB (onnxsim works in memory).
    """

    requires = ("onnx",)
    provides = ("onnx",)
    dependencies = (ONNX,)

    def __init__(self, simplify: bool = True, filename: str = "model.sim.onnx", **base) -> None:
        super().__init__(**base)
        self.simplify = simplify
        self.filename = filename

    def run(self, ctx: ExportContext) -> list[Artifact]:
        onnx = ONNX.load()
        source = ctx.store.require("onnx")
        out = ctx.stage_dir(self.name) / self.filename
        large = source.size_bytes > 1.8e9
        model = onnx.load(str(source.path))
        before = len(model.graph.node)
        model = onnx.shape_inference.infer_shapes(model, data_prop=True) if not large else model
        if self.simplify and ONNXSIM.available and not large:
            simplified, ok = ONNXSIM.load().simplify(model)
            if ok:
                model = simplified
            else:
                logger.warning(
                    "onnxsim could not validate the simplified graph; keeping the original"
                )
        elif self.simplify and not ONNXSIM.available:
            logger.info("onnxsim not installed; shape inference only")
        onnx.save(
            model,
            str(out),
            save_as_external_data=large,
            location=f"{self.filename}.data" if large else None,
        )
        logger.info("ONNX nodes {} -> {} ({})", before, len(model.graph.node), out.name)
        return [
            self.artifact(
                ctx,
                "onnx",
                out,
                nodes_before=before,
                nodes_after=len(model.graph.node),
                **{k: v for k, v in source.metadata.items() if k in ("opset", "inputs", "outputs")},
            )
        ]
