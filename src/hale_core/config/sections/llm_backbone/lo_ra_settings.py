"""``LoRASettings``: Low-rank adapters (Hu et al., 2021, arXiv:2106.09685) on the backbone's linear
layers.
"""

from __future__ import annotations

from pydantic import Field

from hale_core.config.sections.common import StrictModel


class LoRASettings(StrictModel):
    """Low-rank adapters (Hu et al., 2021, arXiv:2106.09685) on the backbone's linear layers.

    ``W x`` becomes ``W x + (alpha / r) * B A x`` with ``A: (r, in)`` and ``B: (out, r)``; only
    ``A`` / ``B`` train, ``W`` stays frozen. ``target_modules`` left empty uses the backbone's
    registered defaults (attention and MLP projections).
    """

    r: int = Field(default=16, ge=1)
    """Adapter rank."""
    alpha: int = Field(default=32, ge=1)
    """Scaling numerator: the update is multiplied by ``alpha / r``."""
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)
    """Dropout on the adapter input."""
    target_modules: list[str] | None = None
    """Module names (suffix match) to adapt; ``None`` = the registered defaults."""
    train_embeddings: bool = False
    """Also train the input embeddings and LM head (needed when new tokens are added and
    their embeddings must be learned, e.g. ``<image>`` / ``<halo_action>``)."""
