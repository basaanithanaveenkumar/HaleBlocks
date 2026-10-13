"""Normalization sub-config."""

from __future__ import annotations

from typing import Literal

from hale_core.config.sections.common import StrictModel


class NormConfig(StrictModel):
    """Normalization layer configuration.

    YAML example::

        model:
          block:
            norm:
              type: rms
              eps: 1e-6
    """

    type: Literal["layer", "rms"] = "layer"
    eps: float = 1e-5
    affine: bool = True
