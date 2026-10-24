"""Shared helpers of :mod:`hale_core.nn.layers.loop` (Components of a looped (weight-tied,
recurrent-depth) transformer).
"""

from __future__ import annotations

from typing import Literal

DepthEmbeddingKind = Literal["none", "sinusoidal", "learned"]


InjectionKind = Literal["none", "add", "concat"]

#: How the ``n`` core layers are scheduled over the passes (Takase & Kiyono, 2021,
#: arXiv:2104.06022): ``cycle`` = ABAB..., ``cycle_rev`` = AB BA AB..., ``sequence`` = AA..BB...
SharingKind = Literal["cycle", "cycle_rev", "sequence"]

#: Per-token compute control: none, Adaptive Computation Time, or Mixture-of-Recursions routing.
HaltingKind = Literal["none", "act", "mor"]
