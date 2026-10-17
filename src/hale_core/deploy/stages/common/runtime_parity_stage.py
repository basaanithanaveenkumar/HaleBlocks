"""Run an exported artifact in its runtime and compare with PyTorch (any backend)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from loguru import logger

from hale_core.deploy.core.artifact import Artifact
from hale_core.deploy.core.export_context import ExportContext
from hale_core.deploy.core.stage import Stage
from hale_core.deploy.exceptions import ParityError, StageError
from hale_core.deploy.registry import RUNTIMES, STAGES
from hale_core.deploy.validation.parity_checker import ParityChecker


@STAGES.register_module()
class RuntimeParityStage(Stage):
    """Generic numerical check: ``RUNTIMES.build(runtime)`` on ``artifact_kind`` vs the reference.

    One stage class serves every backend (Strategy via the ``RUNTIMES`` registry)::

        - {type: RuntimeParityStage, artifact_kind: onnx,
           runtime: {type: OnnxRuntimeSession, providers: [CPUExecutionProvider]}}
        - {type: RuntimeParityStage, artifact_kind: tensorrt_engine,
           runtime: {type: TensorRTSession}}
        - {type: RuntimeParityStage, artifact_kind: mlx_function,
           runtime: {type: MLXFunctionSession}}

    Args:
        artifact_kind: which artifact to load.
        runtime: runtime config (``path`` and, for MLX, ``input_names`` are filled in).
        corners: signature corners to test (``min``/``opt``/``max``) - checks dynamic shapes.
        atol / min_cosine / min_argmax_agreement: see :class:`ParityChecker`.
        cast_inputs: dtype for floating inputs (``float16`` for engines built with fp16 I/O).
    """

    def __init__(
        self,
        artifact_kind: str = "onnx",
        runtime: Mapping[str, Any] | None = None,
        corners: Sequence[str] = ("opt",),
        atol: float = 1e-3,
        min_cosine: float = 0.999,
        min_argmax_agreement: float | None = None,
        cast_inputs: str | None = None,
        **base,
    ) -> None:
        base.setdefault("name", f"{artifact_kind}_parity")
        super().__init__(**base)
        self.artifact_kind = artifact_kind
        self.requires = (artifact_kind,)
        self.runtime_cfg = dict(runtime or {"type": "OnnxRuntimeSession"})
        self.corners = list(corners)
        self.checker = ParityChecker(atol, min_cosine, min_argmax_agreement)
        self.cast_inputs = cast_inputs

    def missing_dependencies(self):
        """Optional packages or executables this stage needs that are not installed."""
        return [d for d in RUNTIMES.get(self.runtime_cfg["type"]).dependencies if not d.available]

    def run(self, ctx: ExportContext) -> list[Artifact]:
        """Execute this stage on the pipeline context (see the class docstring).

        Args:
            ctx: the export context (model, signature, artifacts, work dir).

        Returns:
            The artifacts produced, also recorded in ``ctx.artifacts``.
        """
        artifact = ctx.store.require(self.artifact_kind)
        extra = {"path": str(artifact.path)}
        if self.runtime_cfg["type"] == "MLXFunctionSession":
            extra["input_names"] = ctx.signature.input_names
        session = RUNTIMES.build(self.runtime_cfg, default_args=extra)
        report: dict[str, Any] = {}
        failure: ParityError | None = None
        corners = self.testable_corners(ctx, artifact)
        for corner in corners:
            feeds = {}
            for name, tensor in ctx.sample_inputs(corner).items():
                array = tensor.cpu().numpy()
                feeds[name] = (
                    array.astype(self.cast_inputs)
                    if self.cast_inputs and array.dtype.kind == "f"
                    else array
                )
            outputs = session.run(feeds)
            try:
                report[corner] = self.checker.check(
                    list(ctx.signature.outputs),
                    ctx.reference_outputs(corner),
                    outputs,
                    f"{self.artifact_kind}@{corner}",
                )
            except ParityError as exc:
                report[corner] = exc.metrics
                failure = failure or exc
        path = ctx.stage_dir(self.name) / "parity.json"
        path.write_text(json.dumps(report, indent=2))
        if failure is not None:
            raise failure
        return [self.artifact(ctx, f"{self.artifact_kind}_parity", path, corners=corners)]

    def testable_corners(self, ctx: ExportContext, artifact: Artifact) -> list[str]:
        """Requested corners the artifact can run.

        Statically traced artifacts (an ``.mlxfn`` without ``shapeless``) record the input shapes
        they were traced for in ``traced_shapes``; other corners would fail in the runtime, not
        in the model, so they are skipped with a warning instead of being reported as drift.
        """
        traced = artifact.metadata.get("traced_shapes")
        if not traced:
            return list(self.corners)
        usable = []
        for corner in self.corners:
            shapes = {n: list(t.shape) for n, t in ctx.sample_inputs(corner).items()}
            if shapes in traced:
                usable.append(corner)
            else:
                logger.warning(
                    "{}: corner {!r} {} was not traced into {}; skipped",
                    self.name,
                    corner,
                    shapes,
                    artifact.path.name,
                )
        if not usable:
            raise StageError(
                self.name,
                f"none of the corners {self.corners} were traced into "
                f"{artifact.path.name} (traced: {traced})",
            )
        return usable
