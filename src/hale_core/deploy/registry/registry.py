"""mmdet/mmengine-style registry: string ``type`` in a config dict → class or factory.

A :class:`Registry` maps names to classes (or callables). Components register themselves with a
decorator next to their definition; the pipeline then builds them from config dicts such as::

    {"type": "TensorRTEngineStage", "precision": "fp16"}

Design notes:
    * **Open/closed** - new backends are added by registering a class, never by editing the
      pipeline or the CLI.
    * **Hierarchy** - a child registry (``Registry("stage", parent=STAGES)`` in a downstream
      repo) falls back to its parent on lookup, and a ``scope.Name`` key forces one scope; this is
      how Hale-VLM / HALE-WAM / HaloDiffusionLLM add adapters without touching HaleBlocks.
    * **O(1) lookups** - entries live in a dict; ``build`` copies the config once, so callers'
      dicts are never mutated.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterator, Mapping
from typing import Any

from loguru import logger

from hale_core.deploy.exceptions import RegistryError

BuildFunc = Callable[[Mapping[str, Any], "Registry", dict[str, Any] | None], Any]


class Registry:
    """Name → object table with decorator registration and config-driven building.

    Args:
        name: what is registered (``"stage"``, ``"adapter"`` ...), used in messages.
        parent: optional parent registry consulted when a key is not found locally.
        scope: scope name of this registry (``"hale_core"`` by default, ``"hale_llm"`` in a
            downstream package); ``"scope.Name"`` keys are resolved in that scope only.
        build_func: custom builder; defaults to :meth:`build_from_cfg`.
    """

    def __init__(
        self,
        name: str,
        parent: Registry | None = None,
        scope: str = "hale_core",
        build_func: BuildFunc | None = None,
    ) -> None:
        self.name = name
        self.scope = scope
        self.parent = parent
        self.children: dict[str, Registry] = {}
        self._module_dict: dict[str, Any] = {}
        self._build_func = build_func
        if parent is not None:
            if scope in parent.children:
                raise RegistryError(f"{parent.name} registry already has a child scope {scope!r}")
            parent.children[scope] = self

    # ------------------------------------------------------------------ registration
    def register_module(
        self, name: str | list[str] | None = None, force: bool = False, module: Any = None
    ) -> Any:
        """Register ``module`` under ``name`` (default: its ``__name__``).

        Usable as a decorator (``@STAGES.register_module()``) or a call
        (``STAGES.register_module(module=Cls)``). ``force`` replaces an existing entry.
        """
        if module is not None:
            self._register(module, name, force)
            return module

        def decorator(obj: Any) -> Any:
            self._register(obj, name, force)
            return obj

        return decorator

    def _register(self, module: Any, name: str | list[str] | None, force: bool) -> None:
        if not (inspect.isclass(module) or callable(module)):
            raise RegistryError(
                f"{self.name}: can only register classes or callables, got {type(module)}"
            )
        names = (
            [module.__name__] if name is None else ([name] if isinstance(name, str) else list(name))
        )
        for key in names:
            if key in self._module_dict and not force:
                raise RegistryError(
                    f"{key!r} is already registered in {self.scope}.{self.name}: "
                    f"{self._module_dict[key]}"
                )
            self._module_dict[key] = module
            logger.trace("registered {}.{} {}", self.scope, self.name, key)

    # ------------------------------------------------------------------ lookup
    def get(self, key: str) -> Any:
        """Resolve ``key`` locally, then in the parent chain, then as ``scope.Name``.

        Exact keys are tried first because some registries use dotted names themselves
        (``MLX_OPS`` keys look like ``aten.add.Tensor``); only an unknown dotted key is
        interpreted as a scope-qualified lookup.
        """
        registry: Registry | None = self
        while registry is not None:
            if key in registry._module_dict:
                return registry._module_dict[key]
            registry = registry.parent
        scope, _, short = key.partition(".")
        if short:
            try:
                return self._root().find_scope(scope).get(short)
            except RegistryError:
                pass
        raise RegistryError(
            f"{key!r} is not registered in {self.name}; available: {sorted(self.keys())}"
        )

    def find_scope(self, scope: str) -> Registry:
        """Depth-first search of this registry tree for ``scope``."""
        if self.scope == scope:
            return self
        for child in self.children.values():
            try:
                return child.find_scope(scope)
            except RegistryError:
                continue
        raise RegistryError(f"no {self.name} registry with scope {scope!r}")

    def _root(self) -> Registry:
        registry = self
        while registry.parent is not None:
            registry = registry.parent
        return registry

    def keys(self) -> set[str]:
        """Every resolvable key: local entries plus the parent chain's."""
        own = set(self._module_dict)
        return own | (self.parent.keys() if self.parent is not None else set())

    def qualified_keys(self) -> list[str]:
        """Local keys plus every descendant's keys as ``scope.Name`` (depth-first)."""
        keys = sorted(self._module_dict)
        for scope, child in sorted(self.children.items()):
            keys += [
                k if "." in k and k.split(".")[0] == scope else f"{scope}.{k}"
                for k in child.qualified_keys()
            ]
        return keys

    def __contains__(self, key: str) -> bool:
        try:
            self.get(key)
        except RegistryError:
            return False
        return True

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._module_dict))

    def __len__(self) -> int:
        return len(self._module_dict)

    @property
    def module_dict(self) -> dict[str, Any]:
        """Read-only view of the local entries."""
        return dict(self._module_dict)

    # ------------------------------------------------------------------ building
    def build(self, cfg: Mapping[str, Any], default_args: dict[str, Any] | None = None) -> Any:
        """Instantiate ``cfg["type"]`` with the remaining keys (plus ``default_args``)."""
        func = self._build_func or build_from_cfg
        return func(cfg, self, default_args)

    def __repr__(self) -> str:
        return (
            f"Registry(name={self.name!r}, scope={self.scope!r}, items={sorted(self._module_dict)})"
        )


def build_from_cfg(
    cfg: Mapping[str, Any], registry: Registry, default_args: dict[str, Any] | None = None
) -> Any:
    """Default builder: ``registry.get(cfg["type"])(**rest, **default_args)``.

    Keys in ``cfg`` win over ``default_args``; the input mapping is copied, never mutated.
    """
    if not isinstance(cfg, Mapping):
        raise RegistryError(f"{registry.name} config must be a mapping, got {type(cfg).__name__}")
    if "type" not in cfg:
        raise RegistryError(f"{registry.name} config has no 'type' key: {dict(cfg)}")
    args = dict(default_args or {})
    args.update({k: v for k, v in cfg.items() if k != "type"})
    obj_type = cfg["type"]
    obj_cls = registry.get(obj_type) if isinstance(obj_type, str) else obj_type
    try:
        return obj_cls(**args)
    except TypeError as exc:
        raise RegistryError(
            f"cannot build {registry.name} {obj_type!r} with args {sorted(args)}: {exc}"
        ) from exc
