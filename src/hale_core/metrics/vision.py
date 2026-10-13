"""
Image-level evaluation metrics for HaleBlocks.

These metrics operate directly on model outputs (logits, feature vectors)
and do not depend on any specific backbone architecture.

Metrics
-------
TopKAccuracyMetric    — top-1 and top-5 classification accuracy
LinearProbeMetric     — linear separability of learned features
RetrievalMetric       — nearest-neighbour retrieval R@1, R@5, R@10
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from hale_core.metrics.base import BaseMetric


# ── TopKAccuracyMetric ────────────────────────────────────────────────────────

class TopKAccuracyMetric(BaseMetric):
    """Computes top-1 and top-5 classification accuracy.

    Standard ImageNet evaluation metric.  Top-k accuracy counts a prediction
    as correct if the ground-truth class appears in the k highest-scoring
    classes.

    Usage
    -----
    >>> metric = TopKAccuracyMetric()
    >>> for images, labels in loader:
    ...     logits = backbone_head(images)   # [B, num_classes]
    ...     metric.update(logits=logits, targets=labels)
    >>> print(metric.compute())   # {"acc@1": 76.1, "acc@5": 92.8}
    """

    metric_name = "topk_accuracy"

    def __init__(self, ks: tuple[int, ...] = (1, 5)) -> None:
        """
        Args:
            ks: Which top-k values to compute (default: top-1 and top-5).
        """
        self.ks = ks
        self.reset()

    def reset(self) -> None:
        self._correct: dict[int, int] = {k: 0 for k in self.ks}
        self._total: int = 0

    def update(self, *, logits: Tensor, targets: Tensor) -> None:
        """Accumulate predictions.

        Args:
            logits:  ``[B, C]`` unnormalised class scores.
            targets: ``[B]`` integer ground-truth class indices.
        """
        logits  = logits.detach()
        targets = targets.detach()
        B = targets.shape[0]
        self._total += B

        # top-k predictions: [B, max_k]
        max_k = max(self.ks)
        _, pred = logits.topk(min(max_k, logits.shape[-1]), dim=1, largest=True, sorted=True)
        # pred: [B, k]; expand targets to [B, 1] then compare
        correct = pred.eq(targets.view(-1, 1).expand_as(pred))  # [B, k]

        for k in self.ks:
            # A sample is correct if any of the top-k predictions matches
            self._correct[k] += int(correct[:, :k].any(dim=1).sum().item())

    def compute(self) -> dict[str, float]:
        """Return top-k accuracy as percentages.

        Returns:
            ``acc@k`` for each k in ``self.ks``.  E.g. ``acc@1``, ``acc@5``.
        """
        if self._total == 0:
            return {f"acc@{k}": 0.0 for k in self.ks}
        return {f"acc@{k}": 100.0 * self._correct[k] / self._total for k in self.ks}


# ── LinearProbeMetric ─────────────────────────────────────────────────────────

class LinearProbeMetric(BaseMetric):
    """Evaluates feature quality by fitting a linear probe.

    A linear probe fits a logistic regression classifier on the frozen feature
    vectors.  Good representations are linearly separable — a high probe
    accuracy means the backbone has encoded class-discriminative structure
    without any task-specific fine-tuning.

    Implementation uses scikit-learn ``LogisticRegression`` with an 80/20
    train/test split (seeded for reproducibility).  If sklearn is not
    installed the metric silently returns ``-1.0`` and prints a warning.

    Usage
    -----
    >>> metric = LinearProbeMetric()
    >>> for images, labels in loader:
    ...     out = backbone(images)
    ...     metric.update(features=out.pooled_output, labels=labels)
    >>> print(metric.compute())   # {"linear_probe_acc": 75.4}
    """

    metric_name = "linear_probe"

    def reset(self) -> None:
        self._features: list[Tensor] = []
        self._labels: list[Tensor] = []

    def update(self, *, features: Tensor, labels: Tensor) -> None:
        """Accumulate feature vectors and their class labels.

        Args:
            features: ``[B, D]`` pooled backbone embeddings.
            labels:   ``[B]`` integer class indices.
        """
        self._features.append(features.detach().float().cpu())
        self._labels.append(labels.detach().cpu())

    def compute(self) -> dict[str, float]:
        """Fit logistic regression and return test accuracy.

        Returns:
            linear_probe_acc: Test accuracy (%) on a random 20% hold-out.
                              Returns ``-1.0`` if sklearn is unavailable.
        """
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import StandardScaler
        except ImportError:
            warnings.warn(
                "scikit-learn is not installed; linear_probe_acc will be -1.0. "
                "Install with: pip install scikit-learn",
                RuntimeWarning,
                stacklevel=2,
            )
            return {"linear_probe_acc": -1.0}

        if not self._features:
            return {"linear_probe_acc": 0.0}

        X = torch.cat(self._features, dim=0).numpy()
        y = torch.cat(self._labels, dim=0).numpy()

        # Normalise features for stable logistic regression training
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

        clf = LogisticRegression(
            max_iter=1000,
            C=1.0,
            solver="lbfgs",
            multi_class="auto",
            random_state=42,
        )
        clf.fit(X_tr, y_tr)
        acc = 100.0 * clf.score(X_te, y_te)
        return {"linear_probe_acc": float(acc)}


# ── RetrievalMetric ───────────────────────────────────────────────────────────

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
        query_feats:   Tensor,
        gallery_feats: Tensor,
        query_ids:     Tensor,
        gallery_ids:   Tensor,
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

        Q_feats = torch.cat(self._q_feats, dim=0)   # [Q, D]
        G_feats = torch.cat(self._g_feats, dim=0)   # [G, D]
        Q_ids   = torch.cat(self._q_ids, dim=0)     # [Q]
        G_ids   = torch.cat(self._g_ids, dim=0)     # [G]

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
            retrieved_ids = G_ids[top_k_idx]   # [Q, k]
            # Check if any retrieved item matches the query id
            match = retrieved_ids.eq(Q_ids.unsqueeze(1))  # [Q, k]
            recall = match.any(dim=1).float().mean().item() * 100.0
            results[f"R@{k}"] = float(recall)

        return results
