"""Convert a PyTorch module into an :class:`MLXProgram` via ``torch.export``.

.. code-block:: text

    nn.Module ──torch.export──▶ ExportedProgram ──decompose (keep linear/sdpa/norms)──▶ ATen graph
        ──per-node lookup in MLX_OPS──▶ MLXNode list ──quantize pass──▶ MLXProgram (+ liveness)

Why ``torch.export`` + a converter registry instead of re-implementing every model in MLX:
any model the exporters can trace (all Hale backbones, VLM towers, VLA heads) converts with
no model-specific code, and supporting a new operator is one registered function. High-level
ops (``linear``, ``scaled_dot_product_attention``, ``layer_norm``, ``rms_norm``, ``gelu``,
``silu``) are kept un-decomposed so they map onto MLX's fused kernels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch
from loguru import logger
from torch import fx, nn

from hale_core.deploy.exceptions import ConversionError, RegistryError
from hale_core.deploy.mlx_bridge.dtype_map import to_mlx
from hale_core.deploy.mlx_bridge.mlx_node import MLXNode
from hale_core.deploy.mlx_bridge.mlx_program import MLXProgram
from hale_core.deploy.mlx_bridge.param_ref import ParamRef
from hale_core.deploy.mlx_bridge.value_ref import ValueRef
from hale_core.deploy.registry import MLX_OPS

#: Composite ops kept intact so they hit MLX fused kernels.
PRESERVED_OPS = (
    "linear.default",
    "scaled_dot_product_attention.default",
    "layer_norm.default",
    "rms_norm.default",
    "gelu.default",
    "silu.default",
    "softmax.int",
    "conv2d.default",
    "hardswish.default",
    "adaptive_avg_pool2d.default",
)


class AtenToMLXConverter:
    """Build an :class:`MLXProgram` from a module and example inputs.

    Args:
        dtype: compute dtype of floating weights (``float32`` / ``float16`` / ``bfloat16``).
        quantize: ``{"bits": 4|8, "group_size": 32|64|128}`` - weight-only quantization of
            every linear layer whose input features divide ``group_size`` (others stay dense).
        preserve: override :data:`PRESERVED_OPS`.
    """

    def __init__(
        self,
        dtype: str = "float32",
        quantize: Mapping[str, int] | None = None,
        preserve: Sequence[str] = PRESERVED_OPS,
    ) -> None:
        import hale_core.deploy.mlx_bridge.ops  # noqa: F401 - registers converters (needs mlx)

        self.dtype = dtype
        self.quantize = dict(quantize) if quantize else None
        self.preserve = tuple(preserve)

    # ------------------------------------------------------------------ public
    def convert(
        self, module: nn.Module, example_args: tuple[torch.Tensor, ...], dynamic_shapes: Any = None
    ) -> MLXProgram:
        program = self.export(module, example_args, dynamic_shapes)
        mlx_program = self.lower(program)
        if self.quantize:
            self.quantize_linears(mlx_program)
        logger.info(
            "MLX program: {} nodes, {} params, top ops {}",
            len(mlx_program.nodes),
            len(mlx_program.params),
            list(mlx_program.op_histogram().items())[:6],
        )
        return mlx_program

    def export(
        self, module: nn.Module, example_args, dynamic_shapes=None
    ) -> torch.export.ExportedProgram:
        with torch.no_grad():
            # Deferred runtime asserts: guards the symbolic solver cannot prove (e.g. the mask
            # utilities in transformers) become graph asserts instead of export failures.
            program = torch.export.export(
                module.eval(),
                tuple(example_args),
                dynamic_shapes=dynamic_shapes,
                strict=False,
                prefer_deferred_runtime_asserts_over_guards=True,
            )
        table = torch.export.default_decompositions()
        for name in self.preserve:
            op = getattr(torch.ops.aten, name.split(".")[0], None)
            overload = getattr(op, name.split(".")[1], None) if op is not None else None
            if overload is not None:
                table.pop(overload, None)
        return program.run_decompositions(table)

    # ------------------------------------------------------------------ lowering
    def lower(self, program: torch.export.ExportedProgram) -> MLXProgram:
        sig = program.graph_signature
        tensors = {**program.state_dict, **program.constants}
        params: dict[str, Any] = {}
        placeholders: dict[
            str, Any
        ] = {}  # placeholder name -> ParamRef (weights) or None (user input)
        user_inputs: list[str] = []
        for spec in sig.input_specs:
            arg_name = spec.arg.name
            if spec.kind.name == "USER_INPUT":
                user_inputs.append(arg_name)
                placeholders[arg_name] = ValueRef(arg_name)
            elif spec.target in tensors:
                tensor = tensors[spec.target]
                cast = self.dtype if tensor.is_floating_point() else None
                params[spec.target] = to_mlx(tensor, cast)
                placeholders[arg_name] = ParamRef(spec.target)
            else:
                raise ConversionError(
                    f"input {arg_name} ({spec.kind.name}) has no tensor {spec.target!r}"
                )

        nodes: list[MLXNode] = []
        unsupported: list[str] = []
        outputs: list[Any] = []
        for node in program.graph_module.graph.nodes:
            if node.op == "placeholder":
                continue
            if node.op == "output":
                user = {s.arg.name for s in sig.output_specs if s.kind.name == "USER_OUTPUT"}
                flat = node.args[0] if isinstance(node.args[0], (list, tuple)) else (node.args[0],)
                outputs = [
                    self._arg(a, placeholders)
                    for a in flat
                    if not isinstance(a, fx.Node) or a.name in user
                ]
                continue
            if node.op != "call_function":
                raise ConversionError(f"unexpected FX node kind {node.op!r} ({node.name})")
            key = self.op_key(node.target)
            try:
                fn = MLX_OPS.get(key)
            except RegistryError:
                unsupported.append(key)
                continue
            nodes.append(
                MLXNode(
                    node.name,
                    key,
                    fn,
                    self._arg(tuple(node.args), placeholders),
                    self._arg(dict(node.kwargs), placeholders),
                )
            )
        if unsupported:
            raise ConversionError("the graph cannot be converted to MLX", unsupported)
        return MLXProgram(nodes, params, user_inputs, outputs)

    @staticmethod
    def op_key(target: Any) -> str:
        """Registry key of an FX target.

        ``torch.ops.aten.add.Tensor`` → ``"aten.add.Tensor"``;
        ``operator.mul`` → ``"operator.mul"``.
        """
        if isinstance(target, torch._ops.OpOverload):
            return str(target)
        module = getattr(target, "__module__", "") or ""
        return f"{module.lstrip('_')}.{getattr(target, '__name__', repr(target))}"

    def _arg(self, tree: Any, placeholders: dict[str, Any]) -> Any:
        if isinstance(tree, fx.Node):
            return placeholders.get(tree.name, ValueRef(tree.name))
        if isinstance(tree, tuple):
            return tuple(self._arg(t, placeholders) for t in tree)
        if isinstance(tree, list):
            return [self._arg(t, placeholders) for t in tree]
        if isinstance(tree, dict):
            return {
                k: self._arg(v, placeholders)
                for k, v in tree.items()
                if k not in ("device", "layout", "pin_memory", "memory_format")
            }
        return tree

    # ------------------------------------------------------------------ quantization pass
    def quantize_linears(self, program: MLXProgram) -> int:
        """Rewrite ``linear(x, W, b)`` with a parameter ``W`` into ``quantized_linear``.

        Each weight is quantized once (``mx.quantize``: per-group scales and biases) and the
        dense copy is dropped from the parameter table. Returns the number of rewritten layers.
        """
        import mlx.core as mx

        bits = int(self.quantize.get("bits", 4))
        group = int(self.quantize.get("group_size", 64))
        fn = MLX_OPS.get("hale.quantized_linear")
        done: dict[str, tuple[str, str, str]] = {}
        rewritten = 0
        for node in program.nodes:
            if node.op != "aten.linear.default" or not isinstance(node.args[1], ParamRef):
                continue
            key = node.args[1].key
            weight = program.params.get(key)
            if weight is None or weight.ndim != 2 or weight.shape[1] % group:
                continue
            if key not in done:
                w_q, scales, biases = mx.quantize(weight, group_size=group, bits=bits)
                names = (f"{key}.q", f"{key}.scales", f"{key}.biases")
                program.params.update(zip(names, (w_q, scales, biases), strict=True))
                done[key] = names
            bias = node.args[2] if len(node.args) > 2 else node.kwargs.get("bias")
            node.op, node.fn = "hale.quantized_linear", fn
            node.args = (node.args[0], *(ParamRef(n) for n in done[key]), bias)
            node.kwargs = {"group_size": group, "bits": bits}
            rewritten += 1
        for key in done:  # dense weights no longer referenced
            if not any(
                isinstance(a, ParamRef) and a.key == key for n in program.nodes for a in n.args
            ):
                program.params.pop(key, None)
        logger.info(
            "quantized {} linear layers ({} weights) to {}-bit, group {}",
            rewritten,
            len(done),
            bits,
            group,
        )
        return rewritten
