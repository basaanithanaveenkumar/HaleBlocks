"""LR scheduler sub-config."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from hale_core.config.sections.common import StrictModel


class SchedulerConfig(StrictModel):
    """Learning-rate scheduler configuration.

    YAML examples::

        # No schedule (constant LR — default):
        train:
          scheduler:
            type: constant

        # Cosine with linear warmup (typical LLM setup):
        train:
          scheduler:
            type: warmup_cosine
            warmup_steps: 100
            min_lr_ratio: 0.1

        # One-cycle:
        train:
          scheduler:
            type: one_cycle
            pct_start: 0.3
    """

    type: Literal["constant", "cosine", "linear", "warmup_cosine", "one_cycle"] = "constant"

    # Warmup (applies to warmup_cosine; ignored otherwise)
    warmup_steps: int = 0
    warmup_start_factor: float = 1e-4

    # Decay floor (applies to cosine and warmup_cosine)
    min_lr_ratio: float = 0.1

    # cosine T_max override — None falls back to total_steps from TrainSchedule
    t_max: int | None = None

    # one_cycle
    pct_start: float = 0.3
    div_factor: float = 25.0
    final_div_factor: float = 1e4

    @model_validator(mode="after")
    def _check_warmup(self) -> SchedulerConfig:
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        if not (0.0 < self.min_lr_ratio <= 1.0):
            raise ValueError("min_lr_ratio must be in (0, 1]")
        return self
