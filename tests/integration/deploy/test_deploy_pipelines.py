"""End-to-end deployment pipelines through the public entry point (``DeployRunner``).

* every packaged target config (NVIDIA / Apple / AMD / Qualcomm) must build into a valid
  pipeline - catches broken ``_base_`` chains, typos in stage arguments and data-flow errors;
* the pipelines that can run on any developer machine (ONNX Runtime) or a Mac (MLX) are run
  for real, with numerical parity at the min / opt / max shapes.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path

import pytest
import yaml

from hale_core.deploy import DeployRunner
from hale_core.deploy.config import Config
from hale_core.deploy.core.optional_dependency import OptionalDependency

pytestmark = pytest.mark.integration

REPO = Path(__file__).resolve().parents[3]
PACKAGED = Path(str(resources.files("hale_core.deploy") / "configs"))
TARGETS = sorted(p for p in PACKAGED.rglob("*.yaml") if p.parent.name not in ("_base_", "devices"))
TINY_ADAPTER = {
    "type": "HaleSequenceBackboneAdapter",
    "vocab_size": 128,
    "max_batch": 2,
    "opt_seq": 8,
    "backbone": {
        "arch": "transformer",
        "d_model": 32,
        "n_heads": 2,
        "n_layers": 1,
        "d_ff": 64,
        "max_length": 32,
    },
}


def with_adapter(base: Path, tmp_path: Path, **extra) -> Path:
    cfg = tmp_path / f"{base.stem}.yaml"
    cfg.write_text(
        yaml.safe_dump({"_base_": str(base), "name": base.stem, "adapter": TINY_ADAPTER, **extra})
    )
    return cfg


@pytest.mark.parametrize("target", TARGETS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_every_packaged_target_builds_a_valid_pipeline(target: Path, tmp_path) -> None:
    runner = DeployRunner.from_file(with_adapter(target, tmp_path), work_dir=tmp_path)
    pipeline, ctx = runner.build()
    pipeline.validate()
    assert ctx.device.get("vendor") in {"nvidia", "apple", "amd", "qualcomm"}
    assert len(pipeline.stages) == len(Config.fromfile(target)["pipeline"])


def test_onnx_runtime_pipeline_from_repo_config(tmp_path) -> None:
    report = DeployRunner.from_file(
        REPO / "configs/deploy/gpt_tiny_onnx_ort.yaml", work_dir=tmp_path
    ).run()
    assert report.ok and [r.status for r in report.results] == ["ok", "ok", "ok"]
    run_dir = tmp_path / "gpt_tiny"
    parity = json.loads((run_dir / "onnx_parity/parity.json").read_text())
    assert set(parity) == {"min", "opt", "max"} and all(v["logits"]["ok"] for v in parity.values())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert [a["producer"] for a in manifest["onnx"]] == ["OnnxExportStage", "OnnxSimplifyStage"]
    assert (run_dir / "config.yaml").exists() and (run_dir / "report.json").exists()


def test_vision_tower_to_onnx_runtime_package(tmp_path) -> None:
    cfg = tmp_path / "vision.yaml"
    cfg.write_text(
        yaml.safe_dump(
            {
                "_base_": "pkg://hale_core.deploy/configs/amd/onnxruntime_migraphx_ep.yaml",
                "name": "resnet18",
                "adapter": {
                    "type": "HaleVisionBackboneAdapter",
                    "name": "resnet18",
                    "kwargs": {"img_size": 64},
                    "max_batch": 2,
                },
                "calibration": {"type": "RandomCalibrationSource", "num_batches": 1},
            }
        )
    )
    report = DeployRunner.from_file(cfg, work_dir=tmp_path).run()
    assert report.ok
    deployment = json.loads(
        (tmp_path / "resnet18/OnnxRuntimePackageStage/deployment.json").read_text()
    )
    assert deployment["providers"][0] == "MIGraphXExecutionProvider" and deployment["outputs"] == [
        "pooled_output"
    ]


def test_cross_platform_config_skips_missing_sdks(tmp_path) -> None:
    """A TensorRT config on a machine without TensorRT: the ONNX half runs, the rest is reported."""
    cfg = with_adapter(PACKAGED / "nvidia/tensorrt_fp16.yaml", tmp_path, on_error="continue")
    report = DeployRunner.from_file(cfg, work_dir=tmp_path).run()
    statuses = {r.name: r.status for r in report.results}
    assert statuses["OnnxExportStage"] == statuses["onnx_parity"] == "ok"
    if OptionalDependency("tensorrt").available:
        pytest.skip("TensorRT is installed: the engine is really built here")
    assert (
        statuses["ModelOptAutoCastStage"] == "skipped"
    )  # skip_if_unavailable in the packaged config
    assert statuses["TensorRTEngineStage"] == "failed"
    assert "tensorrt not installed" in report.result("TensorRTEngineStage").message
    assert statuses["tensorrt_engine_parity"] == "skipped"  # its input was never produced


def test_mlx_pipelines_from_repo_configs(tmp_path) -> None:
    pytest.importorskip("mlx.core")
    for name in ("gpt_tiny_mlx_fp32", "gpt_tiny_mlx_int4"):
        report = DeployRunner.from_file(
            REPO / f"configs/deploy/{name}.yaml", work_dir=tmp_path / name
        ).run()
        assert report.ok, report.results
    stage = json.loads(
        (tmp_path / "gpt_tiny_mlx_int4/gpt_tiny/MLXConvertStage/stage.json").read_text()
    )
    assert stage["quantize"] == {"bits": 4, "group_size": 32}
