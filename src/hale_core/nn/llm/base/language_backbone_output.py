"""``LanguageBackboneOutput``: Result of :meth:`LanguageBackbone.forward`."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class LanguageBackboneOutput:
    """Result of :meth:`LanguageBackbone.forward`.

    Attributes:
        logits: ``(B, L, V)`` next-token logits.
        hidden_states: ``(B, L, H)`` final (normalised) hidden states - what action / video
            heads of a VLA read at their query-token positions.
        loss: shifted causal-LM cross-entropy when ``labels`` were given (``-100`` ignored).
    """

    logits: Tensor
    hidden_states: Tensor
    loss: Tensor | None = None
