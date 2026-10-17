"""``FeatureIsotropyMetric``: Measures how isotropic (uniformly spread) the feature distribution is.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from hale_core.metrics.base import BaseMetric

if TYPE_CHECKING:
    pass


class FeatureIsotropyMetric(BaseMetric):
    """Measures how isotropic (uniformly spread) the feature distribution is.

    Why isotropy matters
    --------------------
    Anisotropic feature spaces (features clustered along a few directions) lead
    to poor nearest-neighbour retrieval and biased downstream classifiers.
    Contrastive and self-supervised methods (SimCLR, DINOv2) explicitly optimise
    for isotropy via uniformity losses.

    Isotropy score
    --------------
    Given covariance matrix Σ = (1/N) X^T X:

        isotropy = (1/D) * ||Σ||_F / ||Σ||_*

    where ||·||_F is the Frobenius norm and ||·||_* is the nuclear norm
    (= sum of singular values = trace for PSD Σ).  Score is in (0, 1/D … 1];
    a perfectly isotropic distribution (Σ = σ²I) gives 1.0.

    Usage
    -----
    >>> metric = FeatureIsotropyMetric()
    >>> metric.update(features=out.pooled_output)
    >>> print(metric.compute())
    """

    metric_name = "feature_isotropy"

    def __init__(self, max_samples: int = 4096, n_pairs: int = 1024) -> None:
        """
        Args:
            max_samples: Buffer cap for covariance estimation.
            n_pairs:     Random pairs used for mean cosine similarity.
        """
        self.max_samples = max_samples
        self.n_pairs = n_pairs
        self.reset()

    def reset(self) -> None:
        """Clear the accumulated state before a new evaluation pass."""
        self._buffer: list[Tensor] = []
        self._n_seen: int = 0

    def update(self, *, features: Tensor) -> None:
        """Accumulate feature vectors ``[N, D]``."""
        features = features.detach().float().cpu()
        self._buffer.append(features)
        self._n_seen += features.shape[0]
        if self._n_seen > self.max_samples * 2:
            all_feats = torch.cat(self._buffer, dim=0)
            idx = torch.randperm(all_feats.shape[0])[: self.max_samples]
            self._buffer = [all_feats[idx]]
            self._n_seen = self.max_samples

    def compute(self) -> dict[str, float]:
        """Return isotropy score and mean pairwise cosine similarity.

        Returns:
            isotropy:              Covariance isotropy score in (0, 1].
            mean_cosine_sim:       Mean cosine similarity over random pairs.
                                   Near 0 = spread well; near 1 = collapsed.
        """
        if not self._buffer:
            return {"isotropy": 0.0, "mean_cosine_sim": 0.0}

        X = torch.cat(self._buffer, dim=0).float()  # [N, D]
        X = X - X.mean(dim=0, keepdim=True)
        N, D = X.shape

        # Covariance matrix [D, D]
        cov = (X.T @ X) / max(N - 1, 1)

        # Frobenius norm  = sqrt(sum of squared entries)
        frob = torch.norm(cov, p="fro").item()
        # Nuclear norm  = sum of singular values  = trace for PSD matrix
        nuclear = torch.trace(cov).clamp(min=1e-12).item()

        isotropy = (frob / nuclear) / D

        # Mean cosine similarity over random pairs
        n_pairs = min(self.n_pairs, N * (N - 1) // 2)
        idx_a = torch.randint(0, N, (n_pairs,))
        idx_b = torch.randint(0, N, (n_pairs,))
        # Avoid self-pairs
        same = idx_a == idx_b
        idx_b[same] = (idx_b[same] + 1) % N

        a = X[idx_a]
        b = X[idx_b]
        cos_sim = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()

        return {
            "isotropy": float(isotropy),
            "mean_cosine_sim": float(cos_sim),
        }
