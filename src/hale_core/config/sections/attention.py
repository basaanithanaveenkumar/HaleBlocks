"""Attention sub-config: typed parameters for every registered attention implementation."""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from hale_core.config.sections.common import StrictModel


class AttentionConfig(StrictModel):
    """Fine-grained attention configuration.

    Can be embedded inside :class:`BlockConfig` or :class:`ModelConfig` to give
    precise, implementation-specific control while keeping the top-level
    ``ModelConfig`` fields as convenient shorthands.

    YAML example::

        model:
          block:
            attention:
              impl: gqa
              n_kv_heads: 4
              rope_theta: 500_000
              qk_norm: true
    """

    impl: Literal["mha", "gqa", "mqa"] = "mha"
    n_kv_heads: int | None = None
    rope_theta: float | None = None
    rotary_dim: int | None = None
    qk_norm: bool = False
    window_size: int | None = None
    dropout: float = 0.0

    @model_validator(mode="after")
    def _check_kv_heads(self) -> AttentionConfig:
        if self.impl == "mha" and self.n_kv_heads is not None:
            raise ValueError("n_kv_heads is only valid for gqa / mqa, not mha")
        if self.impl == "mqa" and self.n_kv_heads is not None and self.n_kv_heads != 1:
            raise ValueError("mqa forces n_kv_heads=1; leave it None or set to 1")
        return self
