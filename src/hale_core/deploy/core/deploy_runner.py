"""Facade: config file → adapter, calibration, stages → pipeline run → report on disk."""

from __future__ import annotations

import importlib
import inspect
import random
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from pydantic import ValidationError

from hale_core.deploy.config.config import Config
from hale_core.deploy.config.deploy_config import DeployConfig
from hale_core.deploy.core.export_context import ExportContext
from hale_core.deploy.core.pipeline import Pipeline
from hale_core.deploy.core.pipeline_report import PipelineReport
from hale_core.deploy.exceptions import ConfigError
from hale_core.deploy.registry import ADAPTERS, CALIBRATION, STAGES


class DeployRunner:
    """The single entry point used by the CLI, trainers and tests (Facade pattern).

    Steps: import ``custom_imports`` (repos register adapters/stages) → validate the schema →
    build adapter, calibration source and stages from the registries → validate the pipeline →
    run → write ``config.yaml``, ``manifest.json`` and ``report.json`` into the run directory.
    """

    def __init__(self, config: Config | dict[str, Any], work_dir: str | Path | None = None) -> None:
        self.config = config if isinstance(config, Config) else Config(config)
        self._import_plugins()
        try:
            self.spec = DeployConfig.model_validate(self.config.to_dict())
        except ValidationError as exc:
            raise ConfigError(
                f"invalid deployment config {self.config.filename or ''}:\n{exc}"
            ) from exc
        self.work_dir = Path(work_dir or self.spec.work_dir) / self.spec.name

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        overrides: list[str] | None = None,
        work_dir: str | Path | None = None,
    ) -> DeployRunner:
        return cls(Config.fromfile(path, overrides or []), work_dir)

    def _import_plugins(self) -> None:
        for builtin in (
            "stages",
            "runtimes",
            "calibration",
            "adapters",
        ):  # built-ins register themselves
            importlib.import_module(f"hale_core.deploy.{builtin}")
        for module in self.config.get("custom_imports", []) or []:
            try:
                importlib.import_module(module)
            except ImportError as exc:
                raise ConfigError(f"custom_imports: cannot import {module!r}: {exc}") from exc
            logger.debug("imported plugin {}", module)

    def build(self) -> tuple[Pipeline, ExportContext]:
        torch.manual_seed(self.spec.seed)
        random.seed(self.spec.seed)
        adapter = ADAPTERS.build(self.spec.adapter)
        calibration = CALIBRATION.build(
            self.spec.calibration, default_args=self._calibration_args(adapter)
        )
        stages = [STAGES.build(cfg) for cfg in self.spec.pipeline]
        ctx = ExportContext(
            adapter, calibration, self.work_dir, self.spec.device, self.spec.seed, self.spec.name
        )
        return Pipeline(stages, self.spec.on_error), ctx

    def _calibration_args(self, adapter) -> dict[str, Any]:
        cls = CALIBRATION.get(self.spec.calibration["type"])
        return {"adapter": adapter} if "adapter" in inspect.signature(cls).parameters else {}

    def run(self) -> PipelineReport:
        pipeline, ctx = self.build()
        pipeline.validate()
        self.config.dump(self.work_dir / "config.yaml")
        logger.info(
            "deploying {} → {} ({} stages)", self.spec.name, self.work_dir, len(pipeline.stages)
        )
        try:
            return pipeline.run(ctx)
        finally:
            ctx.store.write_manifest(self.work_dir / "manifest.json")
