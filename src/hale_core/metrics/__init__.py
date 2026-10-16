"""
hale_core.metrics — Evaluation metrics for vision backbone analysis.

All metrics follow the accumulate-then-compute pattern:
    metric.reset()
    for batch in eval_loader:
        metric.update(...)      # keyword args vary per metric
    results = metric.compute()  # dict[str, float]

Backbone metrics (backbone.py)
-------------------------------
ThroughputMetric      — images/sec and per-image latency (mean, p50, p95)
ParameterCountMetric  — total / trainable / frozen parameter counts (millions)
FeatureRankMetric     — effective intrinsic rank via singular-value entropy
FeatureIsotropyMetric — covariance isotropy score + mean cosine similarity
StageActivationMetric — per-stage mean/std/max activation + dead-neuron rate

Image-level metrics (vision.py)
---------------------------------
TopKAccuracyMetric    — top-1 and top-5 classification accuracy
LinearProbeMetric     — linear separability of features (sklearn probe)
RetrievalMetric       — nearest-neighbour retrieval R@1, R@5, R@10
"""

from hale_core.metrics.backbone import (
    FeatureIsotropyMetric,
    FeatureRankMetric,
    ParameterCountMetric,
    StageActivationMetric,
    ThroughputMetric,
)
from hale_core.metrics.base import BaseMetric
from hale_core.metrics.vision import (
    LinearProbeMetric,
    RetrievalMetric,
    TopKAccuracyMetric,
)

__all__ = [
    # Base
    "BaseMetric",
    # Backbone
    "ThroughputMetric",
    "ParameterCountMetric",
    "FeatureRankMetric",
    "FeatureIsotropyMetric",
    "StageActivationMetric",
    # Vision / image-level
    "TopKAccuracyMetric",
    "LinearProbeMetric",
    "RetrievalMetric",
]
