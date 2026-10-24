"""``model.loop`` - how an ``arch: looped`` backbone iterates its shared block."""

from __future__ import annotations

from typing import Any, Literal

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

    ``preset`` names a published recipe (``hale_core.nn.stacks.LOOP_PRESETS``); its options are
    filled in for every field the YAML does not set::

        loop: {preset: mixture_of_recursions, n_loops: 6}
    """

    n_loops: int = 4
    prelude_layers: int = 0
    coda_layers: int = 0
    depth_embedding: Literal["none", "sinusoidal", "learned"] = "sinusoidal"
    input_injection: Literal["none", "add", "concat"] = "add"
    sandwich_norm: bool = False
    halting: Literal["none", "act", "mor"] = "none"
    act_threshold: float = 0.99
    act_init_bias: float = 1.0
    ponder_weight: float = 0.01
    min_loops: int | None = None
    bptt_loops: int | None = None
    exit_tolerance: float | None = None
    scratch_tokens: int = 0
    sharing: Literal["cycle", "cycle_rev", "sequence"] = "cycle"
    lora_rank: int = 0
    mor_balance_weight: float = 0.01
    preset: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _expand_preset(cls, data: Any) -> Any:
        """Fill the preset's options under the explicitly given fields (lazy: torch-free import)."""
        if isinstance(data, dict) and data.get("preset") is not None:
            from hale_core.nn.stacks.looped.loop_presets import resolve_loop_preset

            return resolve_loop_preset(data)
        return data

    @model_validator(mode="after")
    def _check(self) -> LoopConfig:
        from hale_core.nn.stacks.looped import LoopSpec  # lazy: keep config import torch-free

        LoopSpec(**self.model_dump())  # raises ValueError -> ValidationError
        return self
