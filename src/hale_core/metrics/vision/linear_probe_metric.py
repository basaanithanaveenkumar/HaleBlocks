"""``LinearProbeMetric``: Evaluates feature quality by fitting a linear probe."""

from __future__ import annotations

import warnings

import torch
from torch import Tensor

from hale_core.metrics.base import BaseMetric


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
