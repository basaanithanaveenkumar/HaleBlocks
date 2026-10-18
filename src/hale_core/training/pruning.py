"""
In-Training Weight Pruning — PruningManager
============================================

Overview
--------
Pruning removes (zeros out) weights that contribute least to the model's
predictions.  This module implements *in-training* pruning: masks are applied
during the forward pass so the model learns to work around the missing weights.

Three layers of configuration
------------------------------
1. METHOD — which weights to prune:

   magnitude  → Sort all weights by |w|.  Set the bottom k% to zero.
                The cheapest and most commonly used method.  Works well for
                weights with a unimodal distribution (Linear, Conv).

   l1_struct  → Structured L1 pruning: score entire output channels by their
                L1 norm (sum of |w| across the input dimension).  The weakest
                channels are zeroed.  Because whole channels are removed, the
                resulting sub-network is dense and runs fast without sparse
                BLAS kernels.

   random     → Random mask for ablation studies.  Not useful in production.

2. SCHEDULE — when to prune:

   oneshot  → Apply the mask once at start_step.  Fast, but accuracy can drop
              sharply if sparsity is high.  Best for low target_sparsity (≤ 0.3).

   gradual  → Cubic polynomial ramp (Zhu & Gupta, 2018):

                s(t) = s_f + (s_i - s_f) * (1 - (t - t₀) / (t_n - t₀))³

              where s_i=initial_sparsity, s_f=target_sparsity,
                    t₀=start_step, t_n=end_step.

              The model is pruned and re-trained simultaneously, so it adapts
              gracefully.  This is the recommended approach.

3. SCOPE — per-layer vs global threshold:

   global    → A single threshold value is computed across all prunable
               parameters at once.  Some layers may end up sparser than others
               (those with smaller weight magnitudes), but the total sparsity
               is exactly target_sparsity.

   layerwise → Each layer is independently pruned to exactly target_sparsity.
               Simpler, safer, but may prune layers that are already critical
               (e.g. the first layer of a deep network).

How masks work
--------------
PyTorch's torch.nn.utils.prune inserts a "pruning hook" on the module:
  - module.weight_orig  → the original (unmasked) fp32 weights
  - module.weight_mask  → a binary {0, 1} tensor of the same shape
  - module.weight       → weight_orig * weight_mask  (computed on each forward)

The backward pass flows through weight_orig (weight_mask has stop-gradient),
so the optimizer can still update the zeroed weights.  On the next pruning step
the mask is recomputed and new weights may be pruned or unmasked (if gradual
schedule is increasing/decreasing sparsity, though in practice we only increase).

make_permanent() calls torch.nn.utils.prune.remove() which multiplies
weight_orig by weight_mask, writes the result to weight, and removes the hook
and the mask buffer.  The checkpoint is then smaller.

Typical usage
-------------
    pruner = PruningManager(cfg.train.pruning, model)

    # inside training loop, after optimizer.step():
    pruner.step(global_step)        # recompute + apply masks at each interval

    # after training:
    stats = pruner.sparsity_stats() # {"pruning/global_sparsity": 0.48, ...}
    pruner.make_permanent()         # bake masks, remove buffers

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn
import torch.nn.utils.prune as torch_prune
from loguru import logger

if TYPE_CHECKING:
    from hale_core.config.sections.pruning import PruningConfig


class PruningManager:
    """Applies, schedules and tracks weight pruning masks on a model.

    Args:
        cfg:   The PruningConfig section from TrainConfig.
        model: The nn.Module to prune.  Called after .to(device) so the module
               tree is final (no further additions expected).

    The manager is stateless between steps except for:
      - self._prunable_params: list of (module, param_name) that are eligible
      - self._permanent: True once make_permanent() has been called
    """

    def __init__(self, cfg: PruningConfig, model: nn.Module) -> None:
        self.cfg = cfg
        self.model = model
        self._permanent = False

        # Collect prunable (module, param_name) pairs
        self._prunable_params: list[tuple[nn.Module, str]] = []
        if cfg.enabled:
            self._collect_prunable_params()
            logger.info(
                "PruningManager: {} prunable params, method={}, schedule={}, "
                "target_sparsity={:.1%}, scope={}",
                len(self._prunable_params),
                cfg.method,
                cfg.schedule,
                cfg.target_sparsity,
                cfg.scope,
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def step(self, global_step: int) -> None:
        """Call once per optimizer step to apply/update masks.

        For 'oneshot' schedule: applies mask exactly once at start_step.
        For 'gradual' schedule: reapplies mask every prune_interval steps
          between start_step and end_step, with increasing sparsity.

        No-op when:
          - cfg.enabled is False
          - global_step < start_step
          - global_step > end_step (gradual) — mask is held constant
          - make_permanent() has already been called
        """
        if not self.cfg.enabled or self._permanent:
            return

        cfg = self.cfg

        if cfg.schedule == "oneshot":
            if global_step == cfg.start_step:
                self._apply_masks(cfg.target_sparsity)

        elif cfg.schedule == "gradual":
            if global_step < cfg.start_step or global_step > cfg.end_step:
                return
            if (global_step - cfg.start_step) % cfg.prune_interval == 0:
                sparsity = _cubic_sparsity(
                    step=global_step,
                    start_step=cfg.start_step,
                    end_step=cfg.end_step,
                    initial_sparsity=cfg.initial_sparsity,
                    target_sparsity=cfg.target_sparsity,
                )
                self._apply_masks(sparsity)

            # Bake masks at the end of the gradual schedule
            if global_step >= cfg.end_step and cfg.make_permanent:
                self.make_permanent()

    def sparsity_stats(self) -> dict[str, float]:
        """Compute per-layer and global sparsity statistics.

        Returns a dict suitable for logger.log_scalars():
          pruning/global_sparsity  — fraction of ALL prunable weights that are 0
          pruning/<name>_sparsity  — per-layer fraction (for each prunable layer)

        Also logs a summary table at DEBUG level.
        """
        stats: dict[str, float] = {}
        total_zeros = 0
        total_weights = 0

        for module, param_name in self._prunable_params:
            w = getattr(module, param_name)
            zeros = int((w == 0).sum().item())
            total = w.numel()
            sparsity = zeros / max(total, 1)
            # Use the module's registered name via the model's named_modules
            layer_key = _module_key(self.model, module)
            stats[f"pruning/{layer_key}_sparsity"] = sparsity
            total_zeros += zeros
            total_weights += total

        global_sparsity = total_zeros / max(total_weights, 1)
        stats["pruning/global_sparsity"] = global_sparsity
        logger.debug(
            "PruningManager sparsity: {:.1%} global ({}/{} zeros)",
            global_sparsity,
            total_zeros,
            total_weights,
        )
        return stats

    def make_permanent(self) -> None:
        """Bake masks into weights and remove mask buffers.

        After this call:
          - module.weight_orig and module.weight_mask buffers are removed
          - module.weight contains the pruned (sparse) values
          - Checkpoint size is reduced (no mask tensors)
          - The model cannot be further re-pruned without re-calling setup

        Idempotent — safe to call multiple times.
        """
        if self._permanent:
            return
        for module, param_name in self._prunable_params:
            try:
                torch_prune.remove(module, param_name)
            except ValueError:
                logger.debug("prune mask already removed for {}", param_name)
        self._permanent = True
        logger.info("PruningManager: masks baked into weights (permanent)")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _collect_prunable_params(self) -> None:
        """Walk model.named_modules() and collect (module, 'weight') pairs.

        A parameter is prunable if:
          - It is a weight (not bias)
          - The module type is Linear or Conv1d/2d/3d
          - Its full name does NOT match any pattern in cfg.exclude_patterns
        """
        prunable_types = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)
        for name, module in self.model.named_modules():
            if not isinstance(module, prunable_types):
                continue
            if not hasattr(module, "weight") or module.weight is None:
                continue
            if _name_matches_any(name, self.cfg.exclude_patterns):
                logger.debug("PruningManager: skipping {} (excluded)", name)
                continue
            self._prunable_params.append((module, "weight"))

    def _apply_masks(self, sparsity: float) -> None:
        """Apply pruning masks at the given sparsity level.

        Dispatches to the configured method and scope.

        Args:
            sparsity: Fraction of weights to zero (0.0 = no pruning, 1.0 = all).
        """
        if not self._prunable_params:
            return

        amount = max(0.0, min(sparsity, 1.0))  # clamp to [0, 1]

        if self.cfg.scope == "global":
            self._apply_global(amount)
        else:
            self._apply_layerwise(amount)

    def _apply_global(self, amount: float) -> None:
        """One threshold across all prunable params (global scope)."""
        if self.cfg.method == "magnitude":
            torch_prune.global_unstructured(
                self._prunable_params,
                pruning_method=torch_prune.L1Unstructured,
                amount=amount,
            )
        elif self.cfg.method == "l1_struct":
            # Structured global: prune each layer to `amount` by L1 norm of channels
            for module, param_name in self._prunable_params:
                torch_prune.ln_structured(module, name=param_name, amount=amount, n=1, dim=0)
        elif self.cfg.method == "random":
            torch_prune.global_unstructured(
                self._prunable_params,
                pruning_method=torch_prune.RandomUnstructured,
                amount=amount,
            )

    def _apply_layerwise(self, amount: float) -> None:
        """Independent threshold per layer (layerwise scope)."""
        for module, param_name in self._prunable_params:
            if self.cfg.method == "magnitude":
                torch_prune.l1_unstructured(module, name=param_name, amount=amount)
            elif self.cfg.method == "l1_struct":
                torch_prune.ln_structured(module, name=param_name, amount=amount, n=1, dim=0)
            elif self.cfg.method == "random":
                torch_prune.random_unstructured(module, name=param_name, amount=amount)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _cubic_sparsity(
    step: int,
    start_step: int,
    end_step: int,
    initial_sparsity: float,
    target_sparsity: float,
) -> float:
    """Compute the cubic sparsity schedule from Zhu & Gupta (2018).

    Returns current target sparsity in [initial_sparsity, target_sparsity].
    At step <= start_step returns initial_sparsity.
    At step >= end_step returns target_sparsity.
    """
    if end_step <= start_step:
        return target_sparsity
    progress = (step - start_step) / (end_step - start_step)
    progress = max(0.0, min(1.0, progress))
    # s(t) = s_f + (s_i - s_f) * (1 - progress)^3
    cubic_factor = (1.0 - progress) ** 3
    return target_sparsity + (initial_sparsity - target_sparsity) * cubic_factor


def _name_matches_any(name: str, patterns: list[str]) -> bool:
    """Return True if name contains any of the given substrings."""
    return any(pat in name for pat in patterns)


def _module_key(model: nn.Module, target: nn.Module) -> str:
    """Return the dotted name of target within model (e.g. 'layers.0.attn')."""
    for name, module in model.named_modules():
        if module is target:
            return name.replace(".", "_") or "root"
    return "unknown"


def build_pruning_manager(cfg: PruningConfig, model: nn.Module) -> PruningManager:
    """Convenience factory.  Import and call from Trainer.__init__."""
    return PruningManager(cfg, model)
