"""Pipeline validation, error policies, skipping, artifacts and reports."""

from __future__ import annotations

import json

import pytest

from hale_core.deploy.core.artifact import Artifact
from hale_core.deploy.core.artifact_store import ArtifactStore
from hale_core.deploy.core.optional_dependency import OptionalDependency
from hale_core.deploy.core.pipeline import Pipeline
from hale_core.deploy.core.stage import Stage
from hale_core.deploy.exceptions import MissingDependencyError, PipelineError, StageError


class Produce(Stage):
    provides = ("thing",)

    def run(self, ctx):
        path = ctx.stage_dir(self.name) / "thing.txt"
        path.write_text("x")
        return [self.artifact(ctx, "thing", path, n=1)]


class Consume(Stage):
    requires = ("thing",)
    provides = ("other",)

    def run(self, ctx):
        thing = ctx.store.require("thing")
        path = ctx.stage_dir(self.name) / "other.txt"
        path.write_text(thing.path.read_text() * 2)
        return [self.artifact(ctx, "other", path)]


class Explode(Stage):
    provides = ("thing",)

    def run(self, ctx):
        raise RuntimeError("boom")


class NeedsSdk(Stage):
    dependencies = (OptionalDependency("definitely_not_installed_sdk", "pip install nothing"),)

    def run(self, ctx):  # pragma: no cover - never reached
        return []


def test_validation_catches_unsatisfiable_requirements_and_duplicates() -> None:
    with pytest.raises(PipelineError, match="requires"):
        Pipeline([Consume()]).validate()
    with pytest.raises(PipelineError, match="duplicate"):
        Pipeline([Produce(), Produce()])
    with pytest.raises(PipelineError):
        Pipeline([])
    Pipeline([Produce(), Consume()]).validate()
    Pipeline([Consume(enabled=False)]).validate()  # disabled stages are not checked


def test_successful_run_stores_artifacts_and_writes_report(tiny_context) -> None:
    report = Pipeline([Produce(), Consume()]).run(tiny_context)
    assert report.ok and [r.status for r in report.results] == ["ok", "ok"]
    assert tiny_context.store.require("other").path.read_text() == "xx"
    saved = json.loads((tiny_context.work_dir / "report.json").read_text())
    assert saved["ok"] and saved["stages"][1]["name"] == "Consume"


def test_raise_policy_stops_and_continue_policy_records(tiny_context) -> None:
    with pytest.raises(StageError, match="boom"):
        Pipeline([Explode(), Consume()]).run(tiny_context)
    report = Pipeline([Explode(name="e"), Consume()], on_error="continue").run(tiny_context)
    assert [r.status for r in report.results] == ["failed", "skipped"] and not report.ok
    assert "inputs ['thing']" in report.result("Consume").message


def test_missing_dependency_fails_or_skips(tiny_context) -> None:
    report = Pipeline([NeedsSdk(skip_if_unavailable=True)]).run(tiny_context)
    assert (
        report.results[0].status == "skipped" and "pip install nothing" in report.results[0].message
    )
    with pytest.raises(StageError, match="not installed"):
        Pipeline([NeedsSdk()]).run(tiny_context)
    with pytest.raises(MissingDependencyError):
        NeedsSdk.dependencies[0].load()


def test_artifact_store_history_and_manifest(tmp_path) -> None:
    store = ArtifactStore()
    (tmp_path / "a").write_text("1")
    (tmp_path / "b").write_text("22")
    store.put(Artifact("onnx", tmp_path / "a", "s1"))
    store.put(Artifact("onnx", tmp_path / "b", "s2"))
    assert store.require("onnx").producer == "s2" and len(store.history("onnx")) == 2
    with pytest.raises(PipelineError):
        store.require("engine")
    manifest = json.loads(store.write_manifest(tmp_path / "m.json").read_text())
    assert manifest["onnx"][1]["size_bytes"] == 2 and len(manifest["onnx"][0]["sha256"]) == 64
