"""Config loading: _base_ inheritance (files, lists, pkg://), _delete_, cycles, overrides."""

from __future__ import annotations

import json

import pytest
import yaml

from hale_core.deploy.config import Config, ConfigLoader, DictMerger, OverrideParser
from hale_core.deploy.exceptions import ConfigError


def write(path, data) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) if path.suffix == ".json" else yaml.safe_dump(data))
    return str(path)


def test_base_list_order_relative_paths_and_json(tmp_path) -> None:
    write(tmp_path / "bases/a.yaml", {"x": 1, "nested": {"p": 1, "q": 1}, "lst": [1, 2]})
    write(tmp_path / "bases/b.json", {"x": 2, "nested": {"q": 2}})
    top = write(
        tmp_path / "top.yaml",
        {"_base_": ["bases/a.yaml", "bases/b.json"], "nested": {"r": 3}, "lst": [9]},
    )
    cfg = ConfigLoader().load(top)
    assert cfg == {"x": 2, "nested": {"p": 1, "q": 2, "r": 3}, "lst": [9]}


def test_delete_replaces_instead_of_merging(tmp_path) -> None:
    write(tmp_path / "base.yaml", {"stage": {"type": "A", "opt": 1}})
    top = write(
        tmp_path / "top.yaml", {"_base_": "base.yaml", "stage": {"_delete_": True, "type": "B"}}
    )
    assert ConfigLoader().load(top)["stage"] == {"type": "B"}


def test_inheritance_cycle_is_reported(tmp_path) -> None:
    write(tmp_path / "a.yaml", {"_base_": "b.yaml"})
    write(tmp_path / "b.yaml", {"_base_": "a.yaml"})
    with pytest.raises(ConfigError, match="circular"):
        ConfigLoader().load(tmp_path / "a.yaml")


def test_diamond_inheritance_resolves(tmp_path) -> None:
    write(tmp_path / "root.yaml", {"v": 0})
    write(tmp_path / "l.yaml", {"_base_": "root.yaml", "l": 1})
    write(tmp_path / "r.yaml", {"_base_": "root.yaml", "r": 1})
    top = write(tmp_path / "top.yaml", {"_base_": ["l.yaml", "r.yaml"]})
    assert ConfigLoader().load(top) == {"v": 0, "l": 1, "r": 1}


def test_packaged_configs_resolve_through_pkg_scheme(tmp_path) -> None:
    top = write(
        tmp_path / "t.yaml", {"_base_": "pkg://hale_core.deploy/configs/nvidia/tensorrt_fp16.yaml"}
    )
    cfg = ConfigLoader().load(top)
    assert cfg["device"]["vendor"] == "nvidia"
    assert [s["type"] for s in cfg["pipeline"]][-2] == "TensorRTEngineStage"


def test_missing_file_bad_type_and_non_mapping(tmp_path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        ConfigLoader().load(tmp_path / "missing.yaml")
    (tmp_path / "x.toml").write_text("a=1")
    with pytest.raises(ConfigError, match="unsupported"):
        ConfigLoader().load(tmp_path / "x.toml")
    (tmp_path / "list.yaml").write_text("- 1\n- 2\n")
    with pytest.raises(ConfigError, match="mapping"):
        ConfigLoader().load(tmp_path / "list.yaml")
    with pytest.raises(ConfigError, match="not importable"):
        ConfigLoader().locate("pkg://no_such_pkg_xyz/a.yaml", tmp_path)


def test_override_parser_types_and_list_indices() -> None:
    current = {"pipeline": [{"type": "A", "p": 1}, {"type": "B"}], "work_dir": "w"}
    out = OverrideParser().parse(
        ["pipeline.1.precision=int8", "work_dir=out", "seed=3", "flags=[1, 2]", "nested.on=true"],
        current,
    )
    assert out["pipeline"] == [{"type": "A", "p": 1}, {"type": "B", "precision": "int8"}]
    assert out["seed"] == 3 and out["flags"] == [1, 2] and out["nested"] == {"on": True}
    with pytest.raises(ConfigError, match="key.path=value"):
        OverrideParser().parse(["oops"])
    with pytest.raises(ConfigError, match="out of range"):
        OverrideParser().parse(["pipeline.5.x=1"], current)


def test_config_fromfile_overrides_get_path_and_dump(tmp_path) -> None:
    path = write(tmp_path / "c.yaml", {"pipeline": [{"type": "A"}], "name": "m"})
    cfg = Config.fromfile(path, ["pipeline.0.opset=20", "name=other"])
    assert cfg.get_path("pipeline.0.opset") == 20 and cfg["name"] == "other"
    assert cfg.get_path("pipeline.3.type", "dflt") == "dflt"
    assert Config.fromfile(cfg.dump(tmp_path / "out.json")).to_dict() == cfg.to_dict()
    with pytest.raises(ConfigError):
        cfg.dump(tmp_path / "out.txt")


def test_dict_merger_never_mutates_inputs() -> None:
    base, child = {"a": {"b": 1}}, {"a": {"c": 2}}
    merged = DictMerger().merge(base, child)
    assert (
        merged == {"a": {"b": 1, "c": 2}} and base == {"a": {"b": 1}} and child == {"a": {"c": 2}}
    )
