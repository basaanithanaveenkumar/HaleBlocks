"""The export contract of a model: named inputs/outputs and their dynamic dimensions.

Every backend speaks a different dialect of "dynamic shapes"; :class:`IOSignature` is the one
source they are all derived from:

=====================  ==============================================================
consumer               derived form
=====================  ==============================================================
legacy ONNX exporter   ``dynamic_axes={"input_ids": {0: "batch", 1: "seq"}}``
torch.export / dynamo  ``dynamic_shapes={"input_ids": {0: Dim("batch"), ...}}``
TensorRT profiles      ``{"input_ids": ((1, 1), (1, 128), (8, 512))}`` (min/opt/max)
MIGraphX / AI Hub      concrete ``opt`` shapes
calibration / parity   sampled tensors at min / opt / max
=====================  ==============================================================
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from hale_core.deploy.core.dim_spec import DimSpec
from hale_core.deploy.core.tensor_spec import TensorSpec
from hale_core.deploy.exceptions import ConfigError


@dataclass(frozen=True)
class IOSignature:
    """Ordered inputs (positional order of the export module), output names and dims."""

    inputs: tuple[TensorSpec, ...]
    outputs: tuple[str, ...]
    dims: Mapping[str, DimSpec]

    def __post_init__(self) -> None:
        for spec in self.inputs:
            for dim in spec.symbolic_dims.values():
                if dim not in self.dims:
                    raise ConfigError(
                        f"input {spec.name!r} uses undeclared dim {dim!r}; declare it in 'dims'"
                    )
        names = [s.name for s in self.inputs] + list(self.outputs)
        if len(names) != len(set(names)):
            raise ConfigError(f"input/output names must be unique, got {names}")

    @property
    def input_names(self) -> list[str]:
        return [s.name for s in self.inputs]

    def shape(self, spec: TensorSpec, which: str = "opt") -> tuple[int, ...]:
        """Concrete shape of ``spec`` at the ``min`` / ``opt`` / ``max`` corner."""
        return tuple(
            self.dims[d].value(which) if isinstance(d, str) else int(d) for d in spec.shape
        )

    def sample(
        self, which: str = "opt", generator: torch.Generator | None = None
    ) -> dict[str, torch.Tensor]:
        """Random inputs with the right dtypes, shapes and value ranges."""
        out: dict[str, torch.Tensor] = {}
        for spec in self.inputs:
            shape = self.shape(spec, which)
            low, high = spec.value_range
            if spec.torch_dtype == torch.bool:
                out[spec.name] = torch.ones(shape, dtype=torch.bool)
            elif spec.is_integer:
                out[spec.name] = torch.randint(
                    int(low),
                    max(int(high), int(low) + 1),
                    shape,
                    generator=generator,
                    dtype=spec.torch_dtype,
                )
            else:
                t = torch.rand(shape, generator=generator, dtype=torch.float32) * (high - low) + low
                out[spec.name] = t.to(spec.torch_dtype)
        return out

    def as_args(self, tensors: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
        """Positional argument tuple in signature order."""
        return tuple(tensors[name] for name in self.input_names)

    # ------------------------------------------------------------------ backend dialects
    def onnx_dynamic_axes(self) -> dict[str, dict[int, str]]:
        axes = {
            s.name: {i: d for i, d in s.symbolic_dims.items() if self.dims[d].is_dynamic}
            for s in self.inputs
        }
        return {k: v for k, v in axes.items() if v}

    def torch_dynamic_shapes(self) -> tuple[tuple[dict[int, Any] | None, ...]] | None:
        """``dynamic_shapes`` for ``torch.export`` of an :class:`ExportModule`.

        One shared ``torch.export.Dim`` per dim name, so e.g. ``batch`` is the *same* symbol in
        every input (exporters then know the inputs agree). ``ExportModule.forward(*args)``
        receives all inputs as one varargs tuple, hence the extra nesting: ``(specs_per_input,)``.
        """
        dims = {
            name: torch.export.Dim(name, min=d.min, max=d.max)
            for name, d in self.dims.items()
            if d.is_dynamic
        }
        per_input = tuple(
            {i: dims[d] for i, d in s.symbolic_dims.items() if d in dims} or None
            for s in self.inputs
        )
        return (per_input,) if any(per_input) else None

    def tensorrt_profiles(
        self,
    ) -> dict[str, tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
        return {
            s.name: (self.shape(s, "min"), self.shape(s, "opt"), self.shape(s, "max"))
            for s in self.inputs
        }

    @property
    def is_dynamic(self) -> bool:
        return any(d.is_dynamic for d in self.dims.values())

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> IOSignature:
        """``{inputs: [...], outputs: [...], dims: {batch: [1, 1, 8], ...}}``."""
        try:
            dims = {
                name: DimSpec.from_config(name, value)
                for name, value in (cfg.get("dims") or {}).items()
            }
            inputs = tuple(TensorSpec.from_config(i) for i in cfg["inputs"])
            outputs = tuple(o["name"] if isinstance(o, Mapping) else str(o) for o in cfg["outputs"])
        except KeyError as exc:
            raise ConfigError(f"signature config is missing {exc}") from exc
        return cls(inputs, outputs, dims)
