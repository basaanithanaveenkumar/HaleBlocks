"""Shared core API: Registry/NamedRegistry, ConfigLoader inheritance and the exception hierarchy."""

from __future__ import annotations

import pytest

from hale_core.config import load_registered_config  # noqa: F401 - registers the llm schema
from hale_core.config.config import Config
from hale_core.config.config_loader import ConfigLoader
from hale_core.config.load import load_yaml_tree
from hale_core.config.merge import deep_merge
from hale_core.exceptions import ConfigError, HaleCoreError, MissingDependencyError, RegistryError
from hale_core.registry import NamedRegistry, Registry


class _Widget:
    def __init__(self, size: int = 1) -> None:
        self.size = size


def test_exception_hierarchy_keeps_builtin_compatibility():
    assert issubclass(RegistryError, (KeyError, ValueError, HaleCoreError))
    assert issubclass(ConfigError, (ValueError, HaleCoreError))
    err = MissingDependencyError("tensorrt", "pip install tensorrt")
    assert isinstance(err, ImportError) and err.package == "tensorrt" and "pip install" in str(err)
    from hale_core.deploy.exceptions import ConfigError as DeployConfigError
    from hale_core.deploy.exceptions import DeployError

    assert DeployConfigError is ConfigError and issubclass(DeployError, HaleCoreError)


def test_named_registry_is_a_registry_with_the_legacy_interface():
    reg = NamedRegistry("widget")
    assert isinstance(reg, Registry)
    reg.add("b", _Widget)
    reg.add("a", int)
    assert list(reg) == ["b", "a"]                       # registration order, not sorted
    live = reg.items
    reg.add("c", str)
    assert "c" in live                                    # live mapping
    with pytest.raises(ValueError, match="already registered"):
        reg.add("a", float)
    with pytest.raises(KeyError, match="unknown widget"):
        reg.get("zzz")
    assert "zzz" not in reg and "a" in reg
    assert reg.build({"type": "b", "size": 3}).size == 3  # mmdet-style build


def test_registry_scopes_and_build_errors():
    root = Registry("widget")
    child = Registry("widget", parent=root, scope="pkg")
    root.register_module(module=_Widget)
    child.register_module(name="Local", module=_Widget)
    assert child.get("_Widget") is _Widget and root.get("pkg.Local") is _Widget
    with pytest.raises(RegistryError, match="no 'type'"):
        root.build({"size": 1})
    with pytest.raises(RegistryError, match="cannot build"):
        root.build({"type": "_Widget", "colour": "red"})


def test_config_loader_supports_inherits_and_base_at_any_depth(tmp_path):
    (tmp_path / "a.yaml").write_text("model: {width: 64, depth: 2}\nname: a\n")
    (tmp_path / "b.yaml").write_text("inherits: a.yaml\nmodel: {depth: 4}\n")
    (tmp_path / "c.yaml").write_text("inherits: b.yaml\nname: c\n")
    (tmp_path / "d.yaml").write_text("_base_: [c.yaml]\nmodel: {_delete_: true, width: 8}\n")
    assert load_yaml_tree(tmp_path / "c.yaml") == {"model": {"width": 64, "depth": 4}, "name": "c"}
    assert ConfigLoader().load(tmp_path / "d.yaml") == {"model": {"width": 8}, "name": "c"}
    cfg = Config.fromfile(tmp_path / "c.yaml", ["model.depth=12"])
    assert cfg.get_path("model.depth") == 12


def test_config_cycles_and_missing_files_raise(tmp_path):
    (tmp_path / "x.yaml").write_text("inherits: y.yaml\n")
    (tmp_path / "y.yaml").write_text("inherits: x.yaml\n")
    with pytest.raises(ConfigError, match="circular"):
        load_yaml_tree(tmp_path / "x.yaml")
    with pytest.raises(FileNotFoundError):
        load_yaml_tree(tmp_path / "missing.yaml")


def test_deep_merge_drops_inheritance_keys_and_does_not_mutate():
    base = {"a": {"b": 1, "c": 2}}
    out = deep_merge(base, {"inherits": "x.yaml", "a": {"b": 5}})
    assert out == {"a": {"b": 5, "c": 2}} and base == {"a": {"b": 1, "c": 2}}
