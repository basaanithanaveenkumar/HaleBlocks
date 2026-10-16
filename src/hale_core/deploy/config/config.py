"""The merged deployment config as a value object (load, override, dump)."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from hale_core.deploy.config.config_loader import ConfigLoader
from hale_core.deploy.config.dict_merger import DictMerger
from hale_core.deploy.config.override_parser import OverrideParser
from hale_core.deploy.exceptions import ConfigError


class Config(Mapping[str, Any]):
    """Immutable-by-convention view of a merged config dict.

    ``Config.fromfile(path, overrides)`` is the single entry point used by the CLI and by
    repos: it resolves ``_base_`` inheritance, applies ``key.path=value`` overrides and keeps
    the source file for provenance (written next to the exported artifacts).
    """

    def __init__(self, data: Mapping[str, Any] | None = None, filename: str | None = None) -> None:
        self._data: dict[str, Any] = copy.deepcopy(dict(data or {}))
        self.filename = filename

    @classmethod
    def fromfile(cls, path: str | Path, overrides: Iterable[str] = ()) -> Config:
        data = ConfigLoader().load(path)
        cfg = cls(data, filename=str(Path(path).resolve()))
        overrides = list(overrides)
        return cfg.merged(OverrideParser().parse(overrides, cfg._data)) if overrides else cfg

    def merged(self, other: Mapping[str, Any]) -> Config:
        """A new config with ``other`` merged on top (``_delete_`` honoured)."""
        return Config(DictMerger().merge(self._data, other), filename=self.filename)

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """``cfg.get_path("pipeline.0.type")`` - list indices are allowed."""
        node: Any = self._data
        for key in dotted.split("."):
            if isinstance(node, Mapping) and key in node:
                node = node[key]
            elif isinstance(node, list) and key.isdigit() and int(key) < len(node):
                node = node[int(key)]
            else:
                return default
        return node

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def dump(self, path: str | Path) -> Path:
        """Write the merged config as YAML or JSON (by suffix)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            path.write_text(json.dumps(self._data, indent=2, default=str))
        elif path.suffix in (".yaml", ".yml"):
            path.write_text(yaml.safe_dump(self._data, sort_keys=False))
        else:
            raise ConfigError(f"cannot dump config to {path.suffix!r}; use .yaml or .json")
        return path

    def pretty(self) -> str:
        return yaml.safe_dump(self._data, sort_keys=False)

    # Mapping protocol
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"Config(filename={self.filename!r}, keys={list(self._data)})"
