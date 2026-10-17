"""``ParameterCountMetric``: Reports parameter counts for a backbone module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

from hale_core.metrics.base import BaseMetric

if TYPE_CHECKING:
    pass


class ParameterCountMetric(BaseMetric):
    """Reports parameter counts for a backbone module.

    This metric is stateless between ``update`` calls; calling ``update``
    multiple times simply overwrites the previous count (typically only
    called once per model).

    Usage
    -----
    >>> metric = ParameterCountMetric()
    >>> metric.update(backbone=bb)
    >>> print(metric.compute())
    # {"total_params": ..., "trainable_params": ..., "frozen_params": ...}
    """

    metric_name = "parameter_count"

    def reset(self) -> None:
        """Clear the accumulated state before a new evaluation pass."""
        self._total: int = 0
        self._trainable: int = 0

    def update(self, *, backbone: nn.Module) -> None:
        """Count parameters in *backbone*.

        Args:
            backbone: Any ``nn.Module`` — typically a ``BaseVisionBackbone``.
        """
        total = sum(p.numel() for p in backbone.parameters())
        trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        self._total = total
        self._trainable = trainable

    def compute(self) -> dict[str, float]:
        """Return parameter count breakdown.

        Returns:
            total_params:     All parameters (millions).
            trainable_params: Parameters with ``requires_grad=True`` (millions).
            frozen_params:    Parameters with ``requires_grad=False`` (millions).
        """
        frozen = self._total - self._trainable
        M = 1e6
        return {
            "total_params_M": self._total / M,
            "trainable_params_M": self._trainable / M,
            "frozen_params_M": frozen / M,
        }
