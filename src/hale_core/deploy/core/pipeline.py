"""Ordered, validated execution of stages with a configurable error policy."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Literal

from loguru import logger

from hale_core.deploy.core.export_context import ExportContext
from hale_core.deploy.core.pipeline_report import PipelineReport
from hale_core.deploy.core.stage import TORCH_MODULE, Stage
from hale_core.deploy.core.stage_result import StageResult
from hale_core.deploy.exceptions import PipelineError, StageError
from hale_core.exceptions import HaleCoreError


class Pipeline:
    """Run stages in config order after checking that their data flow is satisfiable.

    Validation is a single linear pass over the stages with a set of available artifact kinds
    (O(stages × requirements)), done before any expensive work: a misconfigured pipeline (e.g.
    ``TensorRTEngineStage`` without an ONNX-producing stage before it) fails in milliseconds,
    not after a 20-minute export. Stage names must be unique (they name output directories).

    Args:
        stages: stage instances in execution order.
        on_error: ``raise`` aborts at the first failure; ``continue`` records it and runs every
            later stage whose requirements are still met.
    """

    def __init__(
        self, stages: Sequence[Stage], on_error: Literal["raise", "continue"] = "raise"
    ) -> None:
        if not stages:
            raise PipelineError("a pipeline needs at least one stage")
        names = [s.name for s in stages]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise PipelineError(f"duplicate stage names {duplicates}; set 'name' to disambiguate")
        self.stages = list(stages)
        self.on_error = on_error

    def validate(self, initial: set[str] | None = None) -> None:
        """Check that every stage's required artifacts are produced by an earlier stage.

        Args:
            initial: artifact kinds available before the first stage.

        Raises:
            PipelineError: naming the first stage whose requirements are unmet.
        """
        available = set(initial or {TORCH_MODULE})
        for stage in self.stages:
            if not stage.enabled:
                continue
            missing = [kind for kind in stage.requires if kind not in available]
            if missing:
                raise PipelineError(
                    f"stage {stage.name!r} requires {missing}, which no earlier stage provides "
                    f"(available: {sorted(available)})"
                )
            available.update(stage.provides)

    def run(self, ctx: ExportContext) -> PipelineReport:
        """Run the stages in order and collect their results.

        A failing stage stops the pipeline (or is recorded and skipped when ``on_error`` is
        ``continue``).

        Returns:
            The pipeline report.
        """
        self.validate()
        report = PipelineReport(ctx.work_dir)
        for index, stage in enumerate(self.stages, 1):
            logger.info("[{}/{}] {}", index, len(self.stages), stage.name)
            result = self._run_stage(stage, ctx)
            report.results.append(result)
            if result.status == "failed" and self.on_error == "raise":
                report.log_summary()
                report.write()
                raise StageError(stage.name, result.message)
        report.log_summary()
        report.write()
        return report

    def _run_stage(self, stage: Stage, ctx: ExportContext) -> StageResult:
        kind = type(stage).__name__
        if not stage.enabled:
            return StageResult(stage.name, kind, "skipped", message="disabled")
        unmet = [k for k in stage.requires if k != TORCH_MODULE and ctx.store.get(k) is None]
        if unmet:
            return StageResult(
                stage.name,
                kind,
                "skipped" if self.on_error == "continue" else "failed",
                message=f"inputs {unmet} were not produced",
            )
        missing = stage.missing_dependencies()
        if missing:
            message = "; ".join(
                f"{d.module} not installed ({d.hint})" if d.hint else f"{d.module} not installed"
                for d in missing
            )
            if stage.skip_if_unavailable:
                logger.warning("skipping {}: {}", stage.name, message)
                return StageResult(stage.name, kind, "skipped", message=message)
            return StageResult(stage.name, kind, "failed", message=message)
        start = time.perf_counter()
        try:
            artifacts = stage.run(ctx)
        except HaleCoreError as exc:
            logger.error("{} failed: {}", stage.name, exc)
            return StageResult(
                stage.name, kind, "failed", time.perf_counter() - start, message=str(exc)
            )
        except Exception as exc:  # noqa: BLE001 - vendor SDKs raise anything; keep the run report intact
            logger.exception("{} crashed", stage.name)
            return StageResult(
                stage.name,
                kind,
                "failed",
                time.perf_counter() - start,
                message=f"{type(exc).__name__}: {exc}",
            )
        for artifact in artifacts:
            ctx.store.put(artifact)
        return StageResult(stage.name, kind, "ok", time.perf_counter() - start, artifacts)
