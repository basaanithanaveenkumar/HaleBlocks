"""Shared helpers of :mod:`hale_core.nn.layers.loop` (Components of a looped (weight-tied,
recurrent-depth) transformer).
"""

from __future__ import annotations

from typing import Literal

DepthEmbeddingKind = Literal["none", "sinusoidal", "learned"]


InjectionKind = Literal["none", "add", "concat"]
