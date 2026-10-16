"""``RetrievalMetric``: Nearest-neighbour image retrieval: R@1, R@5, R@10."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from hale_core.metrics.base import BaseMetric


class RetrievalMetric(BaseMetric):
    """Nearest-neighbour image retrieval: R@1, R@5, R@10.

    Recall@k (R@k) is the fraction of queries for which at least one relevant
    item appears in the top-k retrieved results.  Uses cosine similarity.

    This is the standard metric for evaluating visual search quality and
    self-supervised representation learning.

    Usage
    -----
    >>> metric = RetrievalMetric()
    >>> metric.update(
    ...     query_feats=q_feats,     # [Q, D]
    ...     gallery_feats=g_feats,   # [G, D]
    ...     query_ids=q_ids,         # [Q] — class or instance id
    ...     gallery_ids=g_ids,       # [G] — same id space
    ... )
    >>> print(metric.compute())  # {"R@1": ..., "R@5": ..., "R@10": ...}
    """

    metric_name = "retrieval"

    def __init__(self, ks: tuple[int, ...] = (1, 5, 10)) -> None:
        """
        Args:
            ks: Recall@k values to compute.
        """
        self.ks = ks
        self.reset()

    def reset(self) -> None:
        self._q_feats: list[Tensor] = []
        self._g_feats: list[Tensor] = []
        self._q_ids: list[Tensor] = []
        self._g_ids: list[Tensor] = []

    def update(
        self,
        *,
        query_feats: Tensor,
        gallery_feats: Tensor,
        query_ids: Tensor,
        gallery_ids: Tensor,
    ) -> None:
        """Accumulate queries and gallery items.

        Args:
            query_feats:   ``[Q, D]`` query feature vectors.
            gallery_feats: ``[G, D]`` gallery feature vectors.
            query_ids:     ``[Q]`` identifier (class or instance) per query.
            gallery_ids:   ``[G]`` identifier per gallery item.
        """
        self._q_feats.append(query_feats.detach().float().cpu())
        self._g_feats.append(gallery_feats.detach().float().cpu())
        self._q_ids.append(query_ids.detach().cpu())
        self._g_ids.append(gallery_ids.detach().cpu())

    def compute(self) -> dict[str, float]:
        """Compute R@k using cosine similarity.

        Queries are matched against the full accumulated gallery.

        Returns:
            ``R@k`` for each k in ``self.ks`` (as percentages).
        """
        if not self._q_feats:
            return {f"R@{k}": 0.0 for k in self.ks}

        Q_feats = torch.cat(self._q_feats, dim=0)  # [Q, D]
        G_feats = torch.cat(self._g_feats, dim=0)  # [G, D]
        Q_ids = torch.cat(self._q_ids, dim=0)  # [Q]
        G_ids = torch.cat(self._g_ids, dim=0)  # [G]

        # L2-normalise for cosine similarity via matrix multiply
        Q_feats = F.normalize(Q_feats, dim=1)
        G_feats = F.normalize(G_feats, dim=1)

        # Cosine similarity matrix [Q, G]
        sim = Q_feats @ G_feats.T

        results: dict[str, float] = {}
        for k in self.ks:
            # Top-k gallery indices per query [Q, k]
            top_k_idx = sim.topk(min(k, G_feats.shape[0]), dim=1).indices
            # Retrieve their IDs [Q, k]
            retrieved_ids = G_ids[top_k_idx]  # [Q, k]
            # Check if any retrieved item matches the query id
            match = retrieved_ids.eq(Q_ids.unsqueeze(1))  # [Q, k]
            recall = match.any(dim=1).float().mean().item() * 100.0
            results[f"R@{k}"] = float(recall)

        return results
