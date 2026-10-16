"""mmdet-style Registry: registration, lookup, hierarchy, building."""

from __future__ import annotations

import pytest

from hale_core.deploy.exceptions import RegistryError
from hale_core.deploy.registry import Registry


class Widget:
    def __init__(self, size: int = 1, color: str = "red") -> None:
        self.size, self.color = size, color


def test_register_get_and_build_with_default_args() -> None:
    reg = Registry("widget")
    reg.register_module(module=Widget)
    built = reg.build({"type": "Widget", "size": 3}, default_args={"color": "blue", "size": 9})
    assert (built.size, built.color) == (3, "blue")  # cfg keys win over default_args
    assert "Widget" in reg and len(reg) == 1 and list(reg) == ["Widget"]


def test_decorator_aliases_duplicates_and_force() -> None:
    reg = Registry("widget")

    @reg.register_module(name=["a", "b"])
    class A:
        pass

    assert reg.get("a") is reg.get("b") is A
    with pytest.raises(RegistryError, match="already registered"):
        reg.register_module(name="a", module=Widget)
    reg.register_module(name="a", module=Widget, force=True)
    assert reg.get("a") is Widget


def test_child_registry_falls_back_to_parent_and_scope_lookup() -> None:
    root = Registry("widget")
    root.register_module(module=Widget)
    child = Registry("widget", parent=root, scope="downstream")

    @child.register_module()
    class Special:
        pass

    assert child.get("Widget") is Widget  # inherited
    assert root.get("downstream.Special") is Special  # scope-qualified from the root
    with pytest.raises(RegistryError):
        root.get("Special")  # not visible without the scope
    with pytest.raises(RegistryError, match="child scope"):
        Registry("widget", parent=root, scope="downstream")


def test_dotted_keys_are_exact_names_first() -> None:
    ops = Registry("op")
    ops.register_module(name="aten.add.Tensor", module=lambda a, b: a + b)
    assert ops.get("aten.add.Tensor")(1, 2) == 3


def test_build_errors_are_explicit() -> None:
    reg = Registry("widget")
    reg.register_module(module=Widget)
    with pytest.raises(RegistryError, match="no 'type'"):
        reg.build({"size": 1})
    with pytest.raises(RegistryError, match="not registered"):
        reg.build({"type": "Nope"})
    with pytest.raises(RegistryError, match="cannot build"):
        reg.build({"type": "Widget", "weight": 2})
    with pytest.raises(RegistryError, match="mapping"):
        reg.build(["Widget"])  # type: ignore[arg-type]
