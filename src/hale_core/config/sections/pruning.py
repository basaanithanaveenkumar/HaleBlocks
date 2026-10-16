"""
PruningConfig — configuration for in-training weight pruning.

What is pruning?
----------------
Pruning removes weights from a trained (or training) model to reduce its size
and inference cost.  A weight set to zero takes no compute if the underlying
kernel uses sparse arithmetic, or can be physically removed after fine-tuning.

Three axes to configure
------------------------
1. METHOD — which weights to prune:
   "magnitude"  → remove weights with smallest |w|; simplest & most common
   "l1_struct"  → structured L1: prune entire channels/heads ranked by L1 norm;
                   produces dense sub-networks that run fast without sparse kernels
   "random"     → random baseline; useful for ablations

2. SCHEDULE — when to prune:
   "oneshot"    → prune once at `start_step`, then leave masks fixed
   "gradual"    → cubic sparsity schedule from `start_step` to `end_step`;
                   model sees increasing sparsity as training progresses;
                   empirically best for maintaining accuracy
   "iterative"  → prune → fine-tune → prune cycles (manual; managed externally)

3. SCOPE — where to apply masks:
   "global"     → a single threshold is computed across ALL parameters;
                   leads to uneven per-layer sparsity but best overall accuracy
   "layerwise"  → each layer is pruned to `target_sparsity` independently;
                   uniform sparsity; safer for attention projections

Gradual Cubic Schedule
-----------------------
PyTorch's default gradual schedule uses the cubic polynomial proposed in
"To Prune, or Not to Prune" (Zhu & Gupta, 2018):

    s(t) = s_f + (s_i - s_f) * (1 - (t - t_0) / (t_n - t_0))^3

where s_i = initial_sparsity (often 0), s_f = target_sparsity, t_0 = start_step,
t_n = end_step, t = current step.

Field-by-field
--------------
enabled           : master switch
method            : pruning criterion
schedule          : timing strategy
target_sparsity   : final fraction of weights zeroed (0.8 = 80% sparse)
initial_sparsity  : starting sparsity for gradual schedule (usually 0.0)
start_step        : global step at which pruning begins
end_step          : global step at which target_sparsity is reached (gradual only)
prune_interval    : steps between sparsity updates (gradual only)
scope             : global vs per-layer threshold
structured        : if True, prune whole channels/heads (requires l1_struct)
make_permanent    : remove masks and bake zeros after end_step
exclude_patterns  : list of parameter name substrings to skip (e.g. "embedding")
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from hale_core.config.sections.common import StrictModel


class PruningConfig(StrictModel):
    """Controls in-training weight pruning."""

    # ── Master switch ─────────────────────────────────────────────────────────
    enabled: bool = False
    """Set True to activate pruning during training."""

    # ── Pruning criterion ─────────────────────────────────────────────────────
    method: Literal["magnitude", "l1_struct", "random"] = "magnitude"
    """
    magnitude  — prune weights with smallest absolute value; globally or per-layer.
    l1_struct  — prune entire output channels/heads by L1 norm of their weight row.
    random     — random mask (ablation baseline only).
    """

    # ── Schedule ──────────────────────────────────────────────────────────────
    schedule: Literal["oneshot", "gradual"] = "gradual"
    """
    oneshot  — apply mask once at start_step; masks are fixed thereafter.
    gradual  — cubic sparsity ramp from initial_sparsity → target_sparsity
               over start_step … end_step; masks re-applied every prune_interval.
    """

    # ── Sparsity targets ──────────────────────────────────────────────────────
    target_sparsity: float = 0.5
    """Fraction of weights to zero. 0.5 = 50% sparse.  Typical sweet-spots:
    0.3–0.5 for near-lossless; 0.7–0.9 with fine-tuning."""

    initial_sparsity: float = 0.0
    """Starting sparsity for gradual schedule (usually 0.0 = no pruning yet)."""

    # ── Step boundaries ───────────────────────────────────────────────────────
    start_step: int = 1000
    """Global training step to begin pruning.  Wait until the model is partially
    trained so the magnitude rankings are meaningful."""

    end_step: int = 10000
    """Global step at which target_sparsity is fully reached (gradual only).
    After this step, the sparsity mask is held constant."""

    prune_interval: int = 100
    """How often (in steps) to recompute and apply masks during gradual schedule.
    Smaller = smoother sparsity ramp; larger = less overhead."""

    # ── Scope ─────────────────────────────────────────────────────────────────
    scope: Literal["global", "layerwise"] = "global"
    """
    global    — one threshold across all prunable parameters; uneven per-layer.
    layerwise — each layer independently pruned to target_sparsity.
    """

    # ── Structured pruning ────────────────────────────────────────────────────
    structured: bool = False
    """If True and method='l1_struct', prune entire output channels (dim=0)
    instead of individual weights.  Produces dense sub-networks that don't need
    sparse kernels for speedup.  Requires restructuring the next layer too."""

    # ── Post-training ─────────────────────────────────────────────────────────
    make_permanent: bool = True
    """At end_step, convert soft masks to hard zeros and remove the mask buffers,
    reducing checkpoint size.  Set False to keep masks for later analysis."""

    # ── Layer exclusions ──────────────────────────────────────────────────────
    exclude_patterns: list[str] = Field(default_factory=lambda: ["embed", "lm_head", "norm"])
    """Parameter name substrings to skip.  Embeddings and final heads are usually
    excluded because pruning them hurts accuracy disproportionately."""
