"""``hale-deploy`` subcommands and exit codes."""

from __future__ import annotations

import yaml

from hale_core.deploy.cli import main


def test_list_env_and_show(capsys, tmp_path) -> None:
    assert main(["list"]) == 0
    out = capsys.readouterr().out
    assert "TensorRTEngineStage" in out and "MLXConvertStage" in out and "OnnxRuntimeSession" in out
    assert main(["env"]) == 0
    assert "tensorrt" in capsys.readouterr().out
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"_base_": "pkg://hale_core.deploy/configs/apple/mlx_int4.yaml"}))
    assert main(["show", str(cfg), "--cfg-options", "name=demo"]) == 0
    assert "name: demo" in capsys.readouterr().out


def test_config_errors_exit_with_2(tmp_path) -> None:
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(yaml.safe_dump({"pipeline": [{"type": "OnnxExportStage"}]}))  # no adapter
    assert main(["run", str(cfg), "--work-dir", str(tmp_path)]) == 2
    assert main(["show", str(tmp_path / "missing.yaml")]) == 2
