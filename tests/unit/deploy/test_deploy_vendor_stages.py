"""Vendor stages against recording SDK fakes (no GPU / NPU / cloud needed)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml
from fake_sdks import fake_converter_package, fake_migraphx, fake_qai_hub, fake_tensorrt

from hale_core.deploy.core.artifact import Artifact
from hale_core.deploy.exceptions import StageError
from hale_core.deploy.stages.amd.mi_graph_x_compile_stage import MIGraphXCompileStage
from hale_core.deploy.stages.apple.mlx_hugging_face_convert_stage import MLXHuggingFaceConvertStage
from hale_core.deploy.stages.common.onnx_runtime_package_stage import OnnxRuntimePackageStage
from hale_core.deploy.stages.nvidia.model_opt_auto_cast_stage import ModelOptAutoCastStage
from hale_core.deploy.stages.nvidia.tensor_rt_engine_stage import TensorRTEngineStage
from hale_core.deploy.stages.nvidia.tensor_rt_llm_bundle_stage import TensorRTLLMBundleStage
from hale_core.deploy.stages.qualcomm.qai_hub_compile_stage import QAIHubCompileStage


def put_onnx(ctx, **metadata) -> Path:
    path = ctx.stage_dir("export") / "model.onnx"
    path.write_bytes(b"onnx")
    ctx.store.put(Artifact("onnx", path, "export", metadata))
    return path


# ---------------------------------------------------------------------------- TensorRT
def test_tensorrt10_weakly_typed_fp16_with_profiles_and_dla(monkeypatch, tiny_context) -> None:
    trt = fake_tensorrt("10.13.0")
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    put_onnx(tiny_context)
    stage = TensorRTEngineStage(precision="fp16", dla_core=0, hardware_compatibility="ampere_plus")
    (artifact,) = stage.run(tiny_context)
    assert artifact.path.read_bytes() == b"ENGINE" and artifact.metadata["strongly_typed"] is False
    assert "FP16" in trt.calls["flags"] and "GPU_FALLBACK" in trt.calls["flags"]
    assert trt.calls["profiles"]["x"] == ((1, 8), (1, 8), (4, 8))
    assert trt.config.DLA_core == 0 and trt.config.hardware_compatibility_level == "ampere_plus"
    assert trt.calls["network_flags"] == 1  # EXPLICIT_BATCH only
    meta = json.loads((artifact.path.parent / "engine.json").read_text())
    assert meta["profiles"]["x"] == [[1, 8], [1, 8], [4, 8]]


def test_tensorrt10_strongly_typed_when_onnx_is_modelopt_output(monkeypatch, tiny_context) -> None:
    trt = fake_tensorrt("10.13.0")
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    put_onnx(tiny_context, strongly_typed_ready=True)
    TensorRTEngineStage(precision="int8").run(tiny_context)
    assert trt.calls["flags"] == [] and trt.calls["network_flags"] == (1 << 5) | 1


def test_tensorrt11_is_always_strongly_typed(monkeypatch, tiny_context) -> None:
    trt = fake_tensorrt("11.3.0")
    monkeypatch.setitem(sys.modules, "tensorrt", trt)
    put_onnx(tiny_context)
    (artifact,) = TensorRTEngineStage(precision="fp16", strongly_typed=False).run(tiny_context)
    assert artifact.metadata["strongly_typed"] is True and trt.calls["flags"] == []
    assert trt.calls["network_flags"] == 1 << 5


def test_tensorrt_parse_failure_is_reported(monkeypatch, tiny_context) -> None:
    monkeypatch.setitem(sys.modules, "tensorrt", fake_tensorrt(parse_ok=False))
    put_onnx(tiny_context)
    with pytest.raises(StageError, match="unsupported op Foo"):
        TensorRTEngineStage().run(tiny_context)


def test_modelopt_autocast_command_line() -> None:
    cmd = ModelOptAutoCastStage(low_precision_type="bf16", nodes_to_exclude=["/lm_head.*"]).command(
        "in.onnx", "out.onnx"
    )
    assert cmd[1:3] == ["-m", "modelopt.onnx.autocast"] and "--keep_io_types" in cmd
    assert cmd[cmd.index("--low_precision_type") + 1] == "bf16" and "/lm_head.*" in cmd


def test_trtllm_bundle_without_quantization(tiny_context) -> None:
    hf = tiny_context.stage_dir("hf") / "hf"
    hf.mkdir()
    (hf / "config.json").write_text("{}")
    tiny_context.store.put(Artifact("hf_checkpoint", hf, "hf"))
    stage = TensorRTLLMBundleStage(
        max_batch_size=4, llm_api_options={"cuda_graph_config": {"enable": True}}
    )
    assert stage.missing_dependencies() == []
    (artifact,) = stage.run(tiny_context)
    options = yaml.safe_load((artifact.path / "extra_llm_api_options.yaml").read_text())
    assert options["max_batch_size"] == 4 and options["cuda_graph_config"] == {"enable": True}
    assert (artifact.path / "checkpoint/config.json").exists()
    assert "trtllm-serve ./checkpoint --backend pytorch" in (artifact.path / "serve.sh").read_text()


# ---------------------------------------------------------------------------- AMD
def test_migraphx_python_path_quantizes_and_compiles(monkeypatch, tiny_context) -> None:
    migraphx = fake_migraphx()
    monkeypatch.setitem(sys.modules, "migraphx", migraphx)
    monkeypatch.setattr(
        "hale_core.deploy.stages.amd.mi_graph_x_compile_stage.MIGRAPHX",
        type("D", (), {"available": True, "load": staticmethod(lambda: migraphx)})(),
    )
    put_onnx(tiny_context)
    (artifact,) = MIGraphXCompileStage(precision="int8").run(tiny_context)
    assert migraphx.calls["dims"] == {"x": [1, 8]} and migraphx.calls["quantized"] == "int8"
    assert migraphx.calls["calib"] == 2 and migraphx.calls["compile"][0] == "target:gpu"
    assert artifact.path.read_bytes() == b"MXR"


# ---------------------------------------------------------------------------- Qualcomm
def test_qai_hub_options_cover_int64_truncation_and_runtimes() -> None:
    stage = QAIHubCompileStage(
        target_runtime="qnn_dlc", quantize_full_type="w8a16", qairt_version="2.37"
    )
    opts = stage.compile_options(has_int64_io=True)
    assert (
        "--target_runtime qnn_dlc" in opts
        and "--truncate_64bit_io" in opts
        and "--qairt_version 2.37" in opts
    )
    assert "--truncate_64bit_io" not in QAIHubCompileStage(target_runtime="onnx").compile_options(
        True
    )
    assert "--target_runtime" not in QAIHubCompileStage(
        target_runtime="context_binary"
    ).compile_options(False)
    assert QAIHubCompileStage(source="onnx").requires == ("onnx",)
    with pytest.raises(ValueError):
        QAIHubCompileStage(device=None, chipset=None)


def test_qai_hub_compile_profile_flow(monkeypatch, tiny_context) -> None:
    hub = fake_qai_hub(tiny_context.work_dir)
    monkeypatch.setitem(sys.modules, "qai_hub", hub)
    artifacts = QAIHubCompileStage(quantize_full_type="w8a16").run(tiny_context)
    kinds = [a.kind for a in artifacts]
    assert kinds == ["qnn_model", "qnn_profile"] and artifacts[0].path.read_bytes() == b"DLC"
    compile_kwargs = hub.calls["compile"]
    assert compile_kwargs["input_specs"] == {"x": ((1, 8), "float32")}
    assert len(compile_kwargs["calibration_data"]["x"]) == 2  # one entry per calibration batch
    assert (
        json.loads((tiny_context.work_dir / "QAIHubCompileStage/aihub.json").read_text())["job_id"]
        == "j123"
    )


# ---------------------------------------------------------------------------- Apple / packaging
def test_mlx_hf_convert_filters_unknown_kwargs(monkeypatch, tiny_context) -> None:
    pkg = fake_converter_package(("hf_path", "mlx_path", "quantize", "q_bits"))
    monkeypatch.setitem(sys.modules, "mlx_lm", pkg)
    hf = tiny_context.stage_dir("hf")
    tiny_context.store.put(Artifact("hf_checkpoint", hf, "hf"))
    (artifact,) = MLXHuggingFaceConvertStage(q_bits=4).run(tiny_context)
    assert pkg.calls["q_bits"] == 4 and pkg.calls["quantize"] is True and "dtype" not in pkg.calls
    assert artifact.kind == "mlx_checkpoint"


def test_onnx_runtime_package_writes_deployment_manifest(tiny_context) -> None:
    put_onnx(tiny_context)
    stage = OnnxRuntimePackageStage(
        providers=["VitisAIExecutionProvider", "CPUExecutionProvider"],
        provider_options=[{"cache_key": "m"}, {}],
    )
    (artifact,) = stage.run(tiny_context)
    manifest = json.loads((artifact.path / "deployment.json").read_text())
    assert (
        manifest["providers"][0] == "VitisAIExecutionProvider"
        and manifest["dims"]["batch"]["max"] == 4
    )
    assert (artifact.path / "model.onnx").read_bytes() == b"onnx"
