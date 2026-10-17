"""Summary of a pipeline run: per-stage status, timings and artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger

from hale_core.deploy.core.stage_result import StageResult


@dataclass
class PipelineReport:
    """Results of one pipeline run: per-stage status, timing, artifacts and messages."""

    work_dir: Path
    results: list[StageResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether no stage failed (skipped stages are not failures)."""
        return all(r.status != "failed" for r in self.results)

    def result(self, name: str) -> StageResult:
        """Result of the stage named ``name``."""
        for r in self.results:
            if r.name == name:
                return r
        raise KeyError(name)

    def log_summary(self) -> None:
        """Log a one-line-per-stage summary table."""
        logger.info("deployment summary ({})", self.work_dir)
        for r in self.results:
            paths = ", ".join(a.path.name for a in r.artifacts) or "-"
            log = (
                logger.info
                if r.status == "ok"
                else (logger.warning if r.status == "skipped" else logger.error)
            )
            log(
                "  {:<28} {:<8} {:>7.2f}s  {}{}",
                r.name,
                r.status,
                r.seconds,
                paths,
                f"  ({r.message})" if r.message else "",
            )

    def write(self, path: Path | None = None) -> Path:
        """Write the report as JSON (default: ``report.json`` in the work dir) and return the path."""
        path = path or self.work_dir / "report.json"
        path.write_text(
            json.dumps({"ok": self.ok, "stages": [r.to_dict() for r in self.results]}, indent=2)
        )
        return path
