"""A named, possibly dynamic tensor dimension with its min / opt / max range."""

from __future__ import annotations

from dataclasses import dataclass

from hale_core.deploy.exceptions import ConfigError


@dataclass(frozen=True)
class DimSpec:
    """One symbolic dimension (``batch``, ``seq``, ``height`` ...).

    ``opt`` is the size used for tracing and for TensorRT's kernel selection; ``min``/``max``
    bound the range every backend must accept (ONNX dynamic axes, ``torch.export.Dim``,
    TensorRT optimization profiles, MIGraphX dynamic dims). ``min == max`` means static.
    """

    name: str
    min: int
    opt: int
    max: int

    def __post_init__(self) -> None:
        if not 1 <= self.min <= self.opt <= self.max:
            raise ConfigError(
                f"dim {self.name!r} needs 1 <= min <= opt <= max, got "
                f"{self.min}/{self.opt}/{self.max}"
            )

    @property
    def is_dynamic(self) -> bool:
        return self.min != self.max

    @property
    def trace(self) -> int:
        """Size used when *tracing* a dynamic dim: ``opt``, but never 1.

        ``torch.export`` (and the dynamo ONNX exporter built on it) specializes any dimension
        it sees with size 0 or 1 to a constant, which silently makes e.g. ``batch`` static when
        ``opt`` is 1. Tracing with 2 keeps it symbolic; static dims trace at their size.
        """
        if not self.is_dynamic or self.opt > 1:
            return self.opt
        return min(2, self.max)

    def value(self, which: str) -> int:
        """``which`` ∈ {"min", "opt", "max", "trace"}."""
        if which not in ("min", "opt", "max", "trace"):
            raise ValueError(f"which must be min/opt/max/trace, got {which!r}")
        return getattr(self, which)

    @classmethod
    def from_config(cls, name: str, cfg: int | list[int] | dict[str, int]) -> DimSpec:
        """Accept ``8`` (static), ``[1, 8, 32]`` or ``{min: 1, opt: 8, max: 32}``."""
        if isinstance(cfg, int):
            return cls(name, cfg, cfg, cfg)
        if isinstance(cfg, (list, tuple)) and len(cfg) == 3:
            return cls(name, *map(int, cfg))
        if isinstance(cfg, dict):
            return cls(name, int(cfg["min"]), int(cfg["opt"]), int(cfg["max"]))
        raise ConfigError(f"dim {name!r}: expected int, [min, opt, max] or mapping, got {cfg!r}")
