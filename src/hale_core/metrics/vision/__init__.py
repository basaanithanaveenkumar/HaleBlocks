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

from hale_core.metrics.vision.linear_probe_metric import LinearProbeMetric
from hale_core.metrics.vision.retrieval_metric import RetrievalMetric
from hale_core.metrics.vision.top_k_accuracy_metric import TopKAccuracyMetric

__all__ = ["LinearProbeMetric", "RetrievalMetric", "TopKAccuracyMetric"]
