"""``FeatureRankMetric``: Estimates the effective intrinsic dimensionality of feature vectors."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from hale_core.metrics.base import BaseMetric

if TYPE_CHECKING:
    pass


class FeatureRankMetric(BaseMetric):
    """Estimates the effective intrinsic dimensionality of feature vectors.

    Method
    ------
    Accumulates a random reservoir of feature vectors from each ``update`` call
    then at ``compute()`` time performs a truncated SVD.  The effective rank is
    defined as the entropy of the normalised singular-value distribution:

        effective_rank = exp(-Σ p_i * log(p_i))    where p_i = σ_i / Σ σ_j

    This is the *Roy & Vetterli (2007)* definition.  A value of D means the
    features live in a D-dimensional subspace (max possible = embed_dim).

    Usage
    -----
    >>> metric = FeatureRankMetric(max_samples=2048)
    >>> for batch in loader:
    ...     out = backbone(images)
    ...     metric.update(features=out.pooled_output)   # [B, D]
    >>> print(metric.compute())
    """

    metric_name = "feature_rank"

    def __init__(self, max_samples: int = 4096) -> None:
        """
        Args:
            max_samples: Cap on accumulated vectors (reservoir sampling prevents
                         OOM on very long eval loops).
        """
        self.max_samples = max_samples
        self.reset()

    def reset(self) -> None:
        self._buffer: list[Tensor] = []
        self._n_seen: int = 0

    def update(self, *, features: Tensor) -> None:
        """Accumulate feature vectors.

        Args:
            features: ``[N, D]`` float tensor — pooled backbone outputs.
        """
        features = features.detach().float().cpu()
        self._buffer.append(features)
        self._n_seen += features.shape[0]
        # Subsample when buffer exceeds capacity to bound memory
        if self._n_seen > self.max_samples * 2:
            all_feats = torch.cat(self._buffer, dim=0)
            idx = torch.randperm(all_feats.shape[0])[: self.max_samples]
            self._buffer = [all_feats[idx]]
            self._n_seen = self.max_samples

    def compute(self) -> dict[str, float]:
        """Compute effective rank and explained-variance threshold.

        Returns:
            effective_rank:        Roy-Vetterli effective rank (float, ≤ embed_dim).
            explained_variance_90: Minimum k s.t. top-k singular values explain 90% variance.
            explained_variance_99: Same at 99%.
        """
        if not self._buffer:
            return {"effective_rank": 0.0, "explained_variance_90": 0, "explained_variance_99": 0}

        X = torch.cat(self._buffer, dim=0).float()  # [N, D]
        # Centre features (removes DC component from rank estimate)
        X = X - X.mean(dim=0, keepdim=True)

        # Truncated SVD via torch.linalg.svdvals (efficient; no U/V needed)
        sigmas = torch.linalg.svdvals(X)  # [min(N,D)]
        sigmas = sigmas.clamp(min=0)

        # Normalise to a probability distribution over singular values
        total = sigmas.sum().clamp(min=1e-12)
        p = sigmas / total

        # Effective rank: exp(Shannon entropy of p)
        # Avoid log(0) by masking near-zero entries
        nonzero = p > 1e-12
        entropy = -(p[nonzero] * torch.log(p[nonzero])).sum().item()
        effective_rank = math.exp(entropy)

        # Explained variance thresholds
        cumvar = (sigmas**2).cumsum(0) / (sigmas**2).sum().clamp(min=1e-12)
        ev90 = int((cumvar < 0.90).sum().item()) + 1
        ev99 = int((cumvar < 0.99).sum().item()) + 1

        return {
            "effective_rank": effective_rank,
            "explained_variance_90": float(ev90),
            "explained_variance_99": float(ev99),
        }
