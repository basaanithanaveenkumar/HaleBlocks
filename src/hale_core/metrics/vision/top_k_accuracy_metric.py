"""``TopKAccuracyMetric``: Computes top-1 and top-5 classification accuracy."""

from __future__ import annotations

from torch import Tensor

from hale_core.metrics.base import BaseMetric


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
        logits = logits.detach()
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
