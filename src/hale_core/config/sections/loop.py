"""``model.loop`` - how an ``arch: looped`` backbone iterates its shared block."""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from hale_core.config.sections.common import StrictModel


class LoopConfig(StrictModel):
    """YAML mirror of :class:`hale_core.nn.stacks.looped.LoopSpec` (same fields, same checks).

    Ignored unless ``model.arch == "looped"``; ``model.n_layers`` is then the depth of the shared
    core block, and the compute depth is ``prelude_layers + n_loops * n_layers + coda_layers``::

        model:
          arch: looped
          n_layers: 2
          loop:
            n_loops: 6
            halting: act
            ponder_weight: 0.01
    """

    n_loops: int = 4
    prelude_layers: int = 0
    coda_layers: int = 0
    depth_embedding: Literal["none", "sinusoidal", "learned"] = "sinusoidal"
    input_injection: Literal["none", "add", "concat"] = "add"
    sandwich_norm: bool = False
    halting: Literal["none", "act"] = "none"
    act_threshold: float = 0.99
    act_init_bias: float = 1.0
    ponder_weight: float = 0.01
    min_loops: int | None = None
    bptt_loops: int | None = None
    exit_tolerance: float | None = None
    scratch_tokens: int = 0

    @model_validator(mode="after")
    def _check(self) -> LoopConfig:
        from hale_core.nn.stacks.looped import LoopSpec  # lazy: keep config import torch-free

        LoopSpec(**self.model_dump())  # raises ValueError -> ValidationError
        return self
