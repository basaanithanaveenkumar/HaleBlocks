"""
Backbone-level evaluation metrics for HaleBlocks vision backbones.

All classes implement the ``BaseMetric`` interface (reset / update / compute).
They are designed to work with ``BackboneOutput`` produced by any backbone
registered via ``build_vision_backbone()``.

Metrics
-------
ThroughputMetric        — images/second and per-image latency statistics
ParameterCountMetric    — total / trainable / frozen parameter counts
FeatureRankMetric       — effective intrinsic dimensionality via SVD entropy
FeatureIsotropyMetric   — covariance isotropy and mean pairwise cosine sim
StageActivationMetric   — per-stage activation mean / std / dead-neuron rate
"""

from hale_core.metrics.backbone.feature_isotropy_metric import FeatureIsotropyMetric
from hale_core.metrics.backbone.feature_rank_metric import FeatureRankMetric
from hale_core.metrics.backbone.parameter_count_metric import ParameterCountMetric
from hale_core.metrics.backbone.stage_activation_metric import StageActivationMetric
from hale_core.metrics.backbone.throughput_metric import ThroughputMetric

__all__ = [
    "FeatureIsotropyMetric",
    "FeatureRankMetric",
    "ParameterCountMetric",
    "StageActivationMetric",
    "ThroughputMetric",
]
