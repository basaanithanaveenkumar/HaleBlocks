"""Name, symbolic shape and dtype of one model input or output."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from hale_core.deploy.exceptions import ConfigError

#: Config dtype names → torch dtypes (the names are also used for numpy / ONNX / MLX maps).
DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


@dataclass(frozen=True)
class TensorSpec:
    """Shape entries are ints (static) or names of :class:`DimSpec` entries (dynamic).

    Attributes:
        name: graph input/output name (stable across every backend).
        shape: e.g. ``("batch", "seq")`` or ``("batch", 3, 224, 224)``.
        dtype: key of :data:`DTYPES`.
        value_range: ``(low, high)`` for sampled integer inputs (token ids, timesteps), or the
            uniform range of float samples.
    """

    name: str
    shape: tuple[int | str, ...]
    dtype: str = "float32"
    value_range: tuple[float, float] = field(default=(0.0, 1.0))

    def __post_init__(self) -> None:
        if self.dtype not in DTYPES:
            raise ConfigError(
                f"tensor {self.name!r}: unknown dtype {self.dtype!r}; known {sorted(DTYPES)}"
            )

    @property
    def torch_dtype(self) -> torch.dtype:
        """The ``torch.dtype`` of this input."""
        return DTYPES[self.dtype]

    @property
    def is_integer(self) -> bool:
        """Whether the dtype is an integer type (token ids, masks)."""
        return not (self.torch_dtype.is_floating_point or self.torch_dtype == torch.bool)

    @property
    def symbolic_dims(self) -> dict[int, str]:
        """``{axis: dim_name}`` of the dynamic axes."""
        return {i: d for i, d in enumerate(self.shape) if isinstance(d, str)}

    @classmethod
    def from_config(cls, cfg: dict) -> TensorSpec:
        """Build a spec from a config dict (``name``, ``dtype``, ``dims``, value range)."""
        try:
            return cls(
                name=cfg["name"],
                shape=tuple(cfg.get("shape", ())),
                dtype=cfg.get("dtype", "float32"),
                value_range=tuple(cfg.get("value_range", (0.0, 1.0))),
            )
        except KeyError as exc:
            raise ConfigError(f"tensor spec {cfg!r} is missing {exc}") from exc
