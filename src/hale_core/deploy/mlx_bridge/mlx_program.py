"""An executable MLX program converted from a ``torch.export`` graph."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from loguru import logger

from hale_core.deploy.mlx_bridge.mlx_node import MLXNode
from hale_core.deploy.mlx_bridge.param_ref import ParamRef
from hale_core.deploy.mlx_bridge.value_ref import ValueRef


class MLXProgram:
    """Straight-line interpreter over converted nodes (already in topological order).

    Execution keeps an environment ``name → value``. Every node carries the list of values
    whose *last* use it is (computed once by :meth:`compute_liveness`, a reverse scan), and
    those entries are dropped right after the node runs. On long transformer graphs this bounds
    the live set to a few activations instead of every intermediate of the forward pass.

    Weights live in :attr:`params` (resolved via :class:`ParamRef`) so the same program can be
    re-parameterised (e.g. quantized) without touching the node list.

    Args:
        nodes: converted nodes in execution order.
        params: weights / buffers / constants as mlx arrays.
        input_names: names of the user inputs, in positional order.
        outputs: references (or literals) of the user outputs, in order.
    """

    def __init__(
        self,
        nodes: list[MLXNode],
        params: dict[str, Any],
        input_names: Sequence[str],
        outputs: Sequence[Any],
    ) -> None:
        self.nodes = nodes
        self.params = params
        self.input_names = list(input_names)
        self.outputs = list(outputs)
        self.compute_liveness()

    # ------------------------------------------------------------------ analysis
    def compute_liveness(self) -> None:
        """Annotate each node with the values that die after it (one reverse pass, O(edges))."""
        seen: set[str] = {ref.name for ref in self._refs(self.outputs)}
        for node in reversed(self.nodes):
            used = {ref.name for ref in self._refs((node.args, node.kwargs))}
            node.release = tuple(sorted(used - seen))
            seen |= used

    @classmethod
    def _refs(cls, tree: Any):
        if isinstance(tree, ValueRef):
            yield tree
        elif isinstance(tree, (list, tuple)):
            for item in tree:
                yield from cls._refs(item)
        elif isinstance(tree, dict):
            for item in tree.values():
                yield from cls._refs(item)

    # ------------------------------------------------------------------ execution
    def __call__(self, *inputs: Any) -> tuple[Any, ...]:
        if len(inputs) != len(self.input_names):
            raise ValueError(
                f"expected {len(self.input_names)} inputs {self.input_names}, got {len(inputs)}"
            )
        env: dict[str, Any] = dict(zip(self.input_names, inputs, strict=True))
        for node in self.nodes:
            env[node.name] = node.fn(
                *self._resolve(node.args, env), **self._resolve(node.kwargs, env)
            )
            for name in node.release:
                env.pop(name, None)
        return tuple(self._resolve(self.outputs, env))

    def _resolve(self, tree: Any, env: dict[str, Any]) -> Any:
        if isinstance(tree, ValueRef):
            return env[tree.name]
        if isinstance(tree, ParamRef):
            return self.params[tree.key]
        if isinstance(tree, tuple):
            return tuple(self._resolve(t, env) for t in tree)
        if isinstance(tree, list):
            return [self._resolve(t, env) for t in tree]
        if isinstance(tree, dict):
            return {k: self._resolve(v, env) for k, v in tree.items()}
        return tree

    # ------------------------------------------------------------------ persistence
    def op_histogram(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for node in self.nodes:
            counts[node.op] = counts.get(node.op, 0) + 1
        return dict(sorted(counts.items(), key=lambda kv: -kv[1]))

    def save_weights(self, path: Path) -> Path:
        import mlx.core as mx

        mx.save_safetensors(str(path), {k: v for k, v in self.params.items()})
        return path

    def export_function(
        self, path: Path, example_sets: Sequence[Sequence[Any]], shapeless: bool = False
    ) -> Path:
        """Trace the program into a portable ``.mlxfn`` (weights embedded as constants).

        Several example sets (e.g. min / opt / max shapes) are traced into the same file with
        ``mx.exporter`` so the importer can serve each shape without re-tracing.
        """
        import mlx.core as mx

        sets, seen = [], set()
        for example in example_sets:  # identical shapes (e.g. min == opt) must be traced once
            key = tuple((tuple(a.shape), str(a.dtype)) for a in example)
            if key not in seen:
                seen.add(key)
                sets.append(list(example))
        if len(sets) > 1 and hasattr(mx, "exporter"):
            with mx.exporter(str(path), self, shapeless=shapeless) as exporter:
                for args in sets:
                    exporter(*args)
        else:
            mx.export_function(str(path), self, *sets[0], shapeless=shapeless)
        logger.info("exported MLX function {} ({} trace(s))", path, len(sets))
        return path

    def write_manifest(self, path: Path, extra: dict[str, Any] | None = None) -> Path:
        manifest = {
            "inputs": self.input_names,
            "num_nodes": len(self.nodes),
            "num_params": len(self.params),
            "ops": self.op_histogram(),
            **(extra or {}),
        }
        path.write_text(json.dumps(manifest, indent=2, default=str))
        return path
