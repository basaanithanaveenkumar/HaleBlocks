"""Parse ``key.path=value`` overrides (CLI ``--cfg-options``) into a nested dict."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import yaml

from hale_core.exceptions import ConfigError


class OverrideParser:
    """Turn ``["pipeline.1.precision=int8", "work_dir=out"]`` into a mergeable dict.

    * Values are parsed with YAML, so ``8``, ``true``, ``[1, 2]`` and ``{a: 1}`` get their
      natural types and plain words stay strings.
    * A numeric path segment indexes into an existing list (``pipeline.1.precision``); the
      override is applied to a copy of that list so list items can be patched in place.
    """

    def parse(
        self, overrides: Iterable[str], current: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for item in overrides:
            if "=" not in item:
                raise ConfigError(f"override {item!r} must look like key.path=value")
            path, raw = item.split("=", 1)
            keys = [k for k in path.strip().split(".") if k]
            if not keys:
                raise ConfigError(f"override {item!r} has an empty key path")
            self._assign(result, keys, self._value(raw), current or {})
        return result

    @staticmethod
    def _value(raw: str) -> Any:
        try:
            return yaml.safe_load(raw)
        except yaml.YAMLError:
            return raw

    def _assign(self, node: dict[str, Any], keys: list[str], value: Any, current: Any) -> None:
        head, rest = keys[0], keys[1:]
        existing = current.get(head) if isinstance(current, dict) else None
        if not rest:
            node[head] = value
            return
        if isinstance(existing, list) and rest[0].isdigit():
            items = (
                node.get(head)
                if isinstance(node.get(head), list)
                else [dict(x) if isinstance(x, dict) else x for x in existing]
            )
            index = int(rest[0])
            if index >= len(items):
                raise ConfigError(
                    f"override index {index} out of range for {head!r} (len {len(items)})"
                )
            if len(rest) == 1:
                items[index] = value
            else:
                child = items[index] if isinstance(items[index], dict) else {}
                self._assign(child, rest[1:], value, existing[index])
                items[index] = child
            node[head] = items
            return
        child = node.setdefault(head, {})
        if not isinstance(child, dict):
            raise ConfigError(f"override path {'.'.join(keys)!r} descends into a non-mapping")
        self._assign(child, rest, value, existing if isinstance(existing, dict) else {})
