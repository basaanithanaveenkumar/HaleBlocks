"""Convert any traceable model to MLX through the ATen → MLX bridge (Apple Silicon)."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Literal

from hale_core.deploy.core.artifact import Artifact
from hale_core.deploy.core.export_context import ExportContext
from hale_core.deploy.core.optional_dependency import OptionalDependency
from hale_core.deploy.core.stage import Stage
from hale_core.deploy.registry import STAGES

MLX = OptionalDependency("mlx", "pip install mlx (macOS on Apple Silicon)")


@STAGES.register_module()
class MLXConvertStage(Stage):
    """Write ``model.mlxfn`` (portable traced function), ``weights.safetensors`` and a manifest.

    The ``.mlxfn`` runs from Python (``mx.import_function``), C++ and Swift without the original
    model code. Each signature corner in ``export_corners`` is traced into the same file, so
    apps can call it at e.g. batch 1 / short prompt and batch 8 / long prompt.

    Args:
        dtype: compute dtype of floating weights (``float16`` is the usual Metal choice).
        quantize: ``{bits: 4|8, group_size: 32|64|128}`` weight-only quantization of linears.
        export_corners: signature corners traced into the ``.mlxfn``.
        shapeless: trace once with symbolic sizes (only for graphs without shape arithmetic).
    """

    provides = ("mlx_function", "mlx_weights")
    dependencies = (MLX,)

    def __init__(
        self,
        dtype: Literal["float32", "float16", "bfloat16"] = "float16",
        quantize: Mapping[str, int] | None = None,
        export_corners: Sequence[str] = ("opt",),
        shapeless: bool = False,
        **base,
    ) -> None:
        super().__init__(**base)
        self.dtype = dtype
        self.quantize = dict(quantize) if quantize else None
        self.export_corners = list(export_corners)
        self.shapeless = shapeless

    def run(self, ctx: ExportContext) -> list[Artifact]:
        """Execute this stage on the pipeline context (see the class docstring).

        Args:
            ctx: the export context (model, signature, artifacts, work dir).

        Returns:
            The artifacts produced, also recorded in ``ctx.artifacts``.
        """
        from hale_core.deploy.mlx_bridge import AtenToMLXConverter
        from hale_core.deploy.mlx_bridge.dtype_map import to_mlx

        sig = ctx.signature
        converter = AtenToMLXConverter(dtype=self.dtype, quantize=self.quantize)
        program = converter.convert(
            ctx.export_module, ctx.example_args("trace"), sig.torch_dynamic_shapes()
        )
        out = ctx.stage_dir(self.name)
        weights = program.save_weights(out / "weights.safetensors")
        examples = []
        for corner in self.export_corners:
            args = []
            for spec, tensor in zip(sig.inputs, ctx.example_args(corner), strict=True):
                args.append(to_mlx(tensor, self.dtype if tensor.is_floating_point() else None))
            examples.append(args)
        function = program.export_function(out / "model.mlxfn", examples, self.shapeless)
        program.write_manifest(
            out / "mlx_manifest.json",
            {
                "dtype": self.dtype,
                "quantize": self.quantize,
                "corners": self.export_corners,
                "input_specs": [
                    {"name": s.name, "shape": list(s.shape), "dtype": s.dtype} for s in sig.inputs
                ],
                "outputs": list(sig.outputs),
            },
        )
        traced = []  # distinct input-shape sets traced into the .mlxfn (min == opt is traced once)
        for example in examples:
            shapes = {s.name: list(a.shape) for s, a in zip(sig.inputs, example, strict=True)}
            if shapes not in traced:
                traced.append(shapes)
        meta = {
            "dtype": self.dtype,
            "quantize": self.quantize,
            "nodes": len(program.nodes),
            "traced_shapes": None if self.shapeless else traced,
        }
        (out / "stage.json").write_text(json.dumps(meta, indent=2))
        return [
            self.artifact(ctx, "mlx_function", function, **meta),
            self.artifact(ctx, "mlx_weights", weights, **meta),
        ]
