"""Error metrics between a reference output and a backend output."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ParityMetrics:
    """Attributes:
    max_abs: largest absolute difference.
    mean_abs: mean absolute difference.
    cosine: cosine similarity of the flattened outputs (1.0 = same direction); robust for
        low-precision backends where absolute error grows with magnitude.
    argmax_agreement: share of positions whose argmax over the last axis agrees - the metric
        that matters for logits (``None`` for 0-d / 1-d outputs).
    """

    max_abs: float
    mean_abs: float
    cosine: float
    argmax_agreement: float | None

    @classmethod
    def compute(cls, reference: np.ndarray, candidate: np.ndarray) -> ParityMetrics:
        """Max/mean absolute error, cosine similarity and argmax agreement of two arrays."""
        ref = reference.astype(np.float64)
        cand = candidate.astype(np.float64).reshape(ref.shape)
        diff = np.abs(ref - cand)
        denom = np.linalg.norm(ref) * np.linalg.norm(cand)
        cosine = (
            float(np.dot(ref.ravel(), cand.ravel()) / denom)
            if denom > 0
            else float(np.allclose(ref, cand))
        )
        agreement = float(np.mean(ref.argmax(-1) == cand.argmax(-1))) if ref.ndim >= 2 else None
        return cls(
            float(diff.max(initial=0.0)),
            float(diff.mean()) if diff.size else 0.0,
            cosine,
            agreement,
        )

    def to_dict(self) -> dict:
        """The metrics as a plain dict (for reports)."""
        return asdict(self)
