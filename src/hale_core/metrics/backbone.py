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

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from torch import Tensor

from hale_core.metrics.base import BaseMetric

if TYPE_CHECKING:
    pass


# ── ThroughputMetric ──────────────────────────────────────────────────────────

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
            return {k: 0.0 for k in [
                "throughput_img_per_sec", "latency_ms_mean",
                "latency_ms_min", "latency_ms_max", "latency_ms_p50", "latency_ms_p95",
            ]}
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


# ── ParameterCountMetric ──────────────────────────────────────────────────────

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


# ── FeatureRankMetric ─────────────────────────────────────────────────────────

class FeatureRankMetric(BaseMetric):
    """Estimates the effective intrinsic dimensionality of feature vectors.

    Method
    ------
    Accumulates a random reservoir of feature vectors from each ``update`` call
    then at ``compute()`` time performs a truncated SVD.  The effective rank is
    defined as the entropy of the normalised singular-value distribution:

        effective_rank = exp(-Σ p_i * log(p_i))    where p_i = σ_i / Σ σ_j

    This is the *Roy & Vetterli (2007)* definition.  A value of D means the
    features live in a D-dimensional subspace (max possible = embed_dim).

    Usage
    -----
    >>> metric = FeatureRankMetric(max_samples=2048)
    >>> for batch in loader:
    ...     out = backbone(images)
    ...     metric.update(features=out.pooled_output)   # [B, D]
    >>> print(metric.compute())
    """

    metric_name = "feature_rank"

    def __init__(self, max_samples: int = 4096) -> None:
        """
        Args:
            max_samples: Cap on accumulated vectors (reservoir sampling prevents
                         OOM on very long eval loops).
        """
        self.max_samples = max_samples
        self.reset()

    def reset(self) -> None:
        self._buffer: list[Tensor] = []
        self._n_seen: int = 0

    def update(self, *, features: Tensor) -> None:
        """Accumulate feature vectors.

        Args:
            features: ``[N, D]`` float tensor — pooled backbone outputs.
        """
        features = features.detach().float().cpu()
        self._buffer.append(features)
        self._n_seen += features.shape[0]
        # Subsample when buffer exceeds capacity to bound memory
        if self._n_seen > self.max_samples * 2:
            all_feats = torch.cat(self._buffer, dim=0)
            idx = torch.randperm(all_feats.shape[0])[:self.max_samples]
            self._buffer = [all_feats[idx]]
            self._n_seen = self.max_samples

    def compute(self) -> dict[str, float]:
        """Compute effective rank and explained-variance threshold.

        Returns:
            effective_rank:        Roy-Vetterli effective rank (float, ≤ embed_dim).
            explained_variance_90: Minimum k s.t. top-k singular values explain 90% variance.
            explained_variance_99: Same at 99%.
        """
        if not self._buffer:
            return {"effective_rank": 0.0, "explained_variance_90": 0, "explained_variance_99": 0}

        X = torch.cat(self._buffer, dim=0).float()   # [N, D]
        # Centre features (removes DC component from rank estimate)
        X = X - X.mean(dim=0, keepdim=True)

        # Truncated SVD via torch.linalg.svdvals (efficient; no U/V needed)
        sigmas = torch.linalg.svdvals(X)             # [min(N,D)]
        sigmas = sigmas.clamp(min=0)

        # Normalise to a probability distribution over singular values
        total = sigmas.sum().clamp(min=1e-12)
        p = sigmas / total

        # Effective rank: exp(Shannon entropy of p)
        # Avoid log(0) by masking near-zero entries
        nonzero = p > 1e-12
        entropy = -(p[nonzero] * torch.log(p[nonzero])).sum().item()
        effective_rank = math.exp(entropy)

        # Explained variance thresholds
        cumvar = (sigmas ** 2).cumsum(0) / (sigmas ** 2).sum().clamp(min=1e-12)
        ev90 = int((cumvar < 0.90).sum().item()) + 1
        ev99 = int((cumvar < 0.99).sum().item()) + 1

        return {
            "effective_rank": effective_rank,
            "explained_variance_90": float(ev90),
            "explained_variance_99": float(ev99),
        }


# ── FeatureIsotropyMetric ─────────────────────────────────────────────────────

class FeatureIsotropyMetric(BaseMetric):
    """Measures how isotropic (uniformly spread) the feature distribution is.

    Why isotropy matters
    --------------------
    Anisotropic feature spaces (features clustered along a few directions) lead
    to poor nearest-neighbour retrieval and biased downstream classifiers.
    Contrastive and self-supervised methods (SimCLR, DINOv2) explicitly optimise
    for isotropy via uniformity losses.

    Isotropy score
    --------------
    Given covariance matrix Σ = (1/N) X^T X:

        isotropy = (1/D) * ||Σ||_F / ||Σ||_*

    where ||·||_F is the Frobenius norm and ||·||_* is the nuclear norm
    (= sum of singular values = trace for PSD Σ).  Score is in (0, 1/D … 1];
    a perfectly isotropic distribution (Σ = σ²I) gives 1.0.

    Usage
    -----
    >>> metric = FeatureIsotropyMetric()
    >>> metric.update(features=out.pooled_output)
    >>> print(metric.compute())
    """

    metric_name = "feature_isotropy"

    def __init__(self, max_samples: int = 4096, n_pairs: int = 1024) -> None:
        """
        Args:
            max_samples: Buffer cap for covariance estimation.
            n_pairs:     Random pairs used for mean cosine similarity.
        """
        self.max_samples = max_samples
        self.n_pairs = n_pairs
        self.reset()

    def reset(self) -> None:
        self._buffer: list[Tensor] = []
        self._n_seen: int = 0

    def update(self, *, features: Tensor) -> None:
        """Accumulate feature vectors ``[N, D]``."""
        features = features.detach().float().cpu()
        self._buffer.append(features)
        self._n_seen += features.shape[0]
        if self._n_seen > self.max_samples * 2:
            all_feats = torch.cat(self._buffer, dim=0)
            idx = torch.randperm(all_feats.shape[0])[:self.max_samples]
            self._buffer = [all_feats[idx]]
            self._n_seen = self.max_samples

    def compute(self) -> dict[str, float]:
        """Return isotropy score and mean pairwise cosine similarity.

        Returns:
            isotropy:              Covariance isotropy score in (0, 1].
            mean_cosine_sim:       Mean cosine similarity over random pairs.
                                   Near 0 = spread well; near 1 = collapsed.
        """
        if not self._buffer:
            return {"isotropy": 0.0, "mean_cosine_sim": 0.0}

        X = torch.cat(self._buffer, dim=0).float()   # [N, D]
        X = X - X.mean(dim=0, keepdim=True)
        N, D = X.shape

        # Covariance matrix [D, D]
        cov = (X.T @ X) / max(N - 1, 1)

        # Frobenius norm  = sqrt(sum of squared entries)
        frob = torch.norm(cov, p="fro").item()
        # Nuclear norm  = sum of singular values  = trace for PSD matrix
        nuclear = torch.trace(cov).clamp(min=1e-12).item()

        isotropy = (frob / nuclear) / D

        # Mean cosine similarity over random pairs
        n_pairs = min(self.n_pairs, N * (N - 1) // 2)
        idx_a = torch.randint(0, N, (n_pairs,))
        idx_b = torch.randint(0, N, (n_pairs,))
        # Avoid self-pairs
        same = idx_a == idx_b
        idx_b[same] = (idx_b[same] + 1) % N

        a = X[idx_a]
        b = X[idx_b]
        cos_sim = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()

        return {
            "isotropy": float(isotropy),
            "mean_cosine_sim": float(cos_sim),
        }


# ── StageActivationMetric ─────────────────────────────────────────────────────

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
            fm = fm.detach().float()   # [B, C, H, W]
            # Per-channel mean absolute activation [B, C]
            chan_mean_abs = fm.abs().mean(dim=(2, 3))  # [B, C]

            mean_act = float(chan_mean_abs.mean().item())
            std_act  = float(fm.std().item())
            max_abs  = float(fm.abs().max().item())
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
            results[f"{prefix}/mean"]      = sum(self._means[stage_idx]) / len(self._means[stage_idx])
            results[f"{prefix}/std"]       = sum(self._stds[stage_idx]) / len(self._stds[stage_idx])
            results[f"{prefix}/max_abs"]   = max(self._maxabs[stage_idx])
            results[f"{prefix}/dead_frac"] = sum(self._dead_fracs[stage_idx]) / len(self._dead_fracs[stage_idx])
        return results
