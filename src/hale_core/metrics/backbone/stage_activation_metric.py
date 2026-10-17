"""``StageActivationMetric``: Tracks activation statistics across backbone stages."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import Tensor

from hale_core.metrics.base import BaseMetric

if TYPE_CHECKING:
    pass


class StageActivationMetric(BaseMetric):
    """Tracks activation statistics across backbone stages.

    Records per-stage mean absolute activation, standard deviation, peak
    activation magnitude, and dead-neuron fraction.

    Dead neurons
    ------------
    A channel is considered "dead" if its mean absolute activation across
    the spatial dimensions is below a threshold (default 1e-4).  High dead-
    neuron fractions indicate vanishing gradients or over-aggressive pruning.

    Usage
    -----
    >>> metric = StageActivationMetric()
    >>> for batch in loader:
    ...     out = backbone(images)
    ...     metric.update(feature_maps=out.feature_maps)
    >>> stats = metric.compute()
    # {"stage_0/mean": ..., "stage_0/std": ..., "stage_0/dead_frac": ..., ...}
    """

    metric_name = "stage_activations"

    def __init__(self, dead_threshold: float = 1e-4) -> None:
        """
        Args:
            dead_threshold: Channels with mean absolute value below this are
                            counted as dead.
        """
        self.dead_threshold = dead_threshold
        self.reset()

    def reset(self) -> None:
        """Clear the accumulated state before a new evaluation pass."""
        # Per-stage accumulators: lists of scalar stats per batch
        self._means: dict[int, list[float]] = {}
        self._stds: dict[int, list[float]] = {}
        self._maxabs: dict[int, list[float]] = {}
        self._dead_fracs: dict[int, list[float]] = {}

    def update(self, *, feature_maps: list[Tensor]) -> None:
        """Accumulate per-stage statistics.

        Args:
            feature_maps: List of stage tensors ``[B, C, H, W]``, one per stage.
        """
        for stage_idx, fm in enumerate(feature_maps):
            fm = fm.detach().float()  # [B, C, H, W]
            # Per-channel mean absolute activation [B, C]
            chan_mean_abs = fm.abs().mean(dim=(2, 3))  # [B, C]

            mean_act = float(chan_mean_abs.mean().item())
            std_act = float(fm.std().item())
            max_abs = float(fm.abs().max().item())
            # Dead fraction: channels with mean abs < threshold
            dead_frac = float((chan_mean_abs < self.dead_threshold).float().mean().item())

            if stage_idx not in self._means:
                self._means[stage_idx] = []
                self._stds[stage_idx] = []
                self._maxabs[stage_idx] = []
                self._dead_fracs[stage_idx] = []

            self._means[stage_idx].append(mean_act)
            self._stds[stage_idx].append(std_act)
            self._maxabs[stage_idx].append(max_abs)
            self._dead_fracs[stage_idx].append(dead_frac)

    def compute(self) -> dict[str, float]:
        """Return per-stage activation statistics.

        Keys follow the pattern ``stage_{i}/{stat}`` where stat is one of
        ``mean``, ``std``, ``max_abs``, ``dead_frac``.
        """
        results: dict[str, float] = {}
        for stage_idx in sorted(self._means.keys()):
            prefix = f"stage_{stage_idx}"
            results[f"{prefix}/mean"] = sum(self._means[stage_idx]) / len(self._means[stage_idx])
            results[f"{prefix}/std"] = sum(self._stds[stage_idx]) / len(self._stds[stage_idx])
            results[f"{prefix}/max_abs"] = max(self._maxabs[stage_idx])
            results[f"{prefix}/dead_frac"] = sum(self._dead_fracs[stage_idx]) / len(
                self._dead_fracs[stage_idx]
            )
        return results
