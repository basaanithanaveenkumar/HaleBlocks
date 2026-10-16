"""``ThroughputMetric``: Measures inference throughput in images/second and per-image latency."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from hale_core.metrics.base import BaseMetric

if TYPE_CHECKING:
    pass


class ThroughputMetric(BaseMetric):
    """Measures inference throughput in images/second and per-image latency.

    Usage
    -----
    >>> import time
    >>> metric = ThroughputMetric()
    >>> metric.reset()
    >>> t0 = time.perf_counter()
    >>> _ = backbone(x)           # forward pass on a batch of B images
    >>> metric.update(batch_size=B, elapsed_seconds=time.perf_counter() - t0)
    >>> print(metric.compute())
    # {"throughput_img_per_sec": ..., "latency_ms_mean": ..., ...}
    """

    metric_name = "throughput"

    def reset(self) -> None:
        """Clear timing accumulators."""
        self._latencies_ms: list[float] = []  # per-batch latency / batch_size

    def update(self, *, batch_size: int, elapsed_seconds: float) -> None:
        """Record one measured forward pass.

        Args:
            batch_size:       Number of images in the batch.
            elapsed_seconds:  Wall-clock time for the forward pass (seconds).
        """
        per_image_ms = elapsed_seconds * 1000.0 / max(batch_size, 1)
        self._latencies_ms.append(per_image_ms)

    def compute(self) -> dict[str, float]:
        """Return throughput and latency statistics.

        Returns:
            throughput_img_per_sec: Mean images processed per second.
            latency_ms_mean:        Mean per-image latency in milliseconds.
            latency_ms_min:         Best observed per-image latency.
            latency_ms_max:         Worst observed per-image latency.
            latency_ms_p50:         Median per-image latency.
            latency_ms_p95:         95th-percentile per-image latency.
        """
        if not self._latencies_ms:
            return {
                k: 0.0
                for k in [
                    "throughput_img_per_sec",
                    "latency_ms_mean",
                    "latency_ms_min",
                    "latency_ms_max",
                    "latency_ms_p50",
                    "latency_ms_p95",
                ]
            }
        lats = sorted(self._latencies_ms)
        mean_ms = sum(lats) / len(lats)
        n = len(lats)

        def percentile(p: float) -> float:
            idx = int(math.ceil(p / 100.0 * n)) - 1
            return lats[max(0, min(idx, n - 1))]

        return {
            "throughput_img_per_sec": 1000.0 / max(mean_ms, 1e-9),
            "latency_ms_mean": mean_ms,
            "latency_ms_min": lats[0],
            "latency_ms_max": lats[-1],
            "latency_ms_p50": percentile(50),
            "latency_ms_p95": percentile(95),
        }
