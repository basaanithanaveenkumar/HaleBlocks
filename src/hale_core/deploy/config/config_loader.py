"""Load YAML/JSON configs that inherit from base files (``_base_``), mmengine style.

.. code-block:: yaml

    # configs/deploy/gpt_tensorrt_fp16.yaml
    _base_:
      - pkg://hale_core.deploy/configs/nvidia/tensorrt_fp16.yaml   # packaged base
      - ./_shared/gpt_adapter.yaml                                  # relative to this file
    pipeline:
      - type: OnnxExportStage
        opset: 20

Resolution:
    * ``_base_`` is a path or a list of paths; later bases override earlier ones and the file
      itself overrides all its bases (:class:`DictMerger` rules, incl. ``_delete_``).
    * Paths are relative to the including file; ``pkg://package/sub/path.yaml`` resolves inside
      an installed package (``importlib.resources``), so downstream repos can inherit the
      configs shipped in HaleBlocks.
    * Inheritance is a DAG walked depth-first with an explicit *visiting* stack, so cycles are
      reported with their full path instead of recursing forever. Each file is parsed once per
      load (memoised), which keeps diamond-shaped inheritance linear.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from hale_core.deploy.config.dict_merger import DictMerger
from hale_core.deploy.exceptions import ConfigError

BASE_KEY = "_base_"
PKG_SCHEME = "pkg://"
SUFFIXES = {".yaml", ".yml", ".json"}


class ConfigLoader:
    """Resolve a config file and all its ``_base_`` ancestors into one plain dict."""

    def __init__(self, merger: DictMerger | None = None) -> None:
        self.merger = merger or DictMerger()

    def load(self, path: str | Path) -> dict[str, Any]:
        """Return the fully merged config of ``path``."""
        cache: dict[Path, dict[str, Any]] = {}
        return self._resolve(self.locate(str(path), Path.cwd()), [], cache)

    # ------------------------------------------------------------------ resolution
    def _resolve(
        self, path: Path, visiting: list[Path], cache: dict[Path, dict[str, Any]]
    ) -> dict[str, Any]:
        if path in visiting:
            chain = " -> ".join(str(p) for p in [*visiting, path])
            raise ConfigError(f"circular _base_ inheritance: {chain}")
        if path in cache:
            return cache[path]
        raw = self.read(path)
        bases = raw.pop(BASE_KEY, [])
        bases = [bases] if isinstance(bases, str) else list(bases)
        merged: dict[str, Any] = {}
        for base in bases:
            base_path = self.locate(base, path.parent)
            logger.debug("config {} inherits {}", path.name, base_path)
            merged = self.merger.merge(merged, self._resolve(base_path, [*visiting, path], cache))
        result = self.merger.merge(merged, raw)
        cache[path] = result
        return result

    def locate(self, ref: str, relative_to: Path) -> Path:
        """Turn a ``_base_`` reference into an absolute file path."""
        if ref.startswith(PKG_SCHEME):
            package, _, sub = ref[len(PKG_SCHEME) :].partition("/")
            try:
                root = resources.files(package)
            except ModuleNotFoundError as exc:
                raise ConfigError(f"package {package!r} of {ref!r} is not importable") from exc
            path = Path(str(root.joinpath(sub)))
        else:
            path = Path(ref).expanduser()
            if not path.is_absolute():
                path = relative_to / path
        path = path.resolve()
        if not path.is_file():
            raise ConfigError(f"config file not found: {ref} (resolved to {path})")
        if path.suffix not in SUFFIXES:
            raise ConfigError(
                f"unsupported config type {path.suffix!r} for {path}; use {sorted(SUFFIXES)}"
            )
        return path

    @staticmethod
    def read(path: Path) -> dict[str, Any]:
        """Parse one YAML or JSON file (no inheritance)."""
        try:
            text = path.read_text()
            data = json.loads(text) if path.suffix == ".json" else yaml.safe_load(text)
        except (OSError, json.JSONDecodeError, yaml.YAMLError) as exc:
            raise ConfigError(f"cannot parse config {path}: {exc}") from exc
        if data is None:
            return {}
        if not isinstance(data, dict):
            raise ConfigError(
                f"config {path} must be a mapping at the top level, got {type(data).__name__}"
            )
        return data
