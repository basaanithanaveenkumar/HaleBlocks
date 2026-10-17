"""Variants and registrations of :mod:`hale_core.nn.layers.dit` (DiT block (Peebles & Xie): pre-norm
attention + FFN with AdaLN-Zero).
"""

from __future__ import annotations

from hale_core.nn.layers.dit.di_t_block import DiTBlock


def _register_dit_block() -> None:
    from hale_core.registry.plugins import BLOCKS_REGISTRY

    if "dit" not in BLOCKS_REGISTRY:
        BLOCKS_REGISTRY.add("dit", DiTBlock)


_register_dit_block()
