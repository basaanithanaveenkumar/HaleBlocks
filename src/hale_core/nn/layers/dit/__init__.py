"""DiT block (Peebles & Xie): pre-norm attention + FFN with AdaLN-Zero.

Zero-init modulation so the block starts as a residual transformer; gates, scales,
and shifts are learned from the timestep embedding.
"""

import hale_core.nn.layers.dit._variants  # noqa: F401 - module-level registrations
from hale_core.nn.layers.dit.di_t_block import DiTBlock
from hale_core.nn.layers.dit.di_t_final_layer import DiTFinalLayer

__all__ = ["DiTBlock", "DiTFinalLayer"]
