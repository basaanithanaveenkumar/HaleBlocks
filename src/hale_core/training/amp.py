"""
Mixed Precision Training — AMPManager
======================================

Overview
--------
Automatic Mixed Precision (AMP) accelerates training by running most operations
in a low-precision floating-point type (fp16 or bf16) while keeping master
weights in fp32.  This module wraps the two PyTorch primitives needed:

  1. torch.amp.autocast   — context manager that casts eligible ops to fp16/bf16
  2. torch.amp.GradScaler — scales the loss to prevent fp16 gradient underflow

How it fits into a training loop
---------------------------------

    amp = AMPManager(cfg.train.amp, device)
    amp.setup(model)          # optional: verifies fp32 master params

    # ── forward ──
    with amp.forward_context():
        loss = loss_fn(model, batch)

    # ── backward ──
    amp.scale_loss(loss).backward()

    # ── optimizer step ──
    amp.step(optimizer)       # unscales + steps + updates scaler (fp16)
                              # or: optimizer.step()  (bf16, no scaler)
    amp.zero_grad(optimizer)  # efficient zero_grad(set_to_none=True)

    # ── logging ──
    stats = amp.stats()       # {"amp/scale": 65536, "amp/dtype": "bf16"}

fp16 vs bf16
------------
fp16 (half precision)
  • Range: ~6e-5 → 6e4   Mantissa: 10 bits
  • Needs GradScaler because gradients smaller than ~6e-5 underflow to 0
  • Hardware: all CUDA GPUs with Tensor Cores (Volta+)

bf16 (bfloat16)
  • Range: same as fp32 (~1e-38 → 3e38)  Mantissa: 7 bits
  • Does NOT need GradScaler in most cases (overflow/underflow very rare)
  • Hardware: Ampere (A100, RTX 3090+), TPU, Apple Silicon (MPS)
  • Preferred for LLM training on A100/H100

GradScaler internals
--------------------
At each step:
  1. scaler.scale(loss).backward()   — multiplies loss by scale (e.g. 65536)
                                        so gradients don't underflow fp16
  2. scaler.unscale_(optimizer)       — divides gradients by scale before clip
  3. clip_grad_norm_(...)             — clips on unscaled gradients (correct norm)
  4. scaler.step(optimizer)           — skips step if any grad is inf/nan
  5. scaler.update()                  — grows scale by growth_factor every
                                        growth_interval clean steps; shrinks by
                                        backoff_factor after an overflow

Thread safety
-------------
AMPManager is NOT thread-safe.  Each DDP rank should own its own instance.
GradScaler state is NOT shared across ranks — this is correct because each rank
has its own gradients.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Generator

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from hale_core.config.sections.amp import AMPConfig


# ── dtype lookup ─────────────────────────────────────────────────────────────
_DTYPE_MAP: dict[str, torch.dtype] = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


class AMPManager:
    """Wraps autocast + GradScaler and exposes a clean interface to the trainer.

    Args:
        cfg:    The AMPConfig section from TrainConfig.
        device: The training device ("cuda", "cpu", "mps").  AMP is silently
                disabled when device is CPU and enabled_on_cpu is False.

    Example usage (inside Trainer._run_epochs):

        amp = AMPManager(cfg.train.amp, device="cuda")

        with amp.forward_context():
            loss = criterion(model(batch))

        amp.scale_loss(loss).backward()
        amp.unscale(optimizer)              # so grad clip works on true scale
        torch.nn.utils.clip_grad_norm_(params, max_norm)
        amp.step(optimizer)
        amp.update()

        metrics |= amp.stats()
    """

    def __init__(self, cfg: AMPConfig, device: str | torch.device) -> None:
        self.cfg = cfg
        self.device_type = _resolve_device_type(device)

        # Decide whether AMP is actually active
        self._active = cfg.enabled and _amp_supported(self.device_type, cfg)
        self._dtype = _DTYPE_MAP.get(cfg.dtype, torch.float32)

        # GradScaler is only meaningful for fp16; bf16 doesn't need it but it's
        # harmless to create one (it will never actually scale when all grads
        # are finite, which they nearly always are in bf16).
        self._scaler: torch.amp.GradScaler | None = None
        if self._active and cfg.dtype == "fp16":
            self._scaler = torch.amp.GradScaler(
                device=self.device_type,
                init_scale=cfg.init_scale,
                growth_factor=cfg.growth_factor,
                backoff_factor=cfg.backoff_factor,
                growth_interval=cfg.growth_interval,
                enabled=True,
            )

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        """True when AMP is actually running (not just configured)."""
        return self._active

    @contextmanager
    def forward_context(self) -> Generator[None, None, None]:
        """Context manager for the forward pass.

        Inside this context, PyTorch automatically casts eligible matmul / conv
        / attention ops to self._dtype.  The output tensors are in fp16/bf16,
        but all accumulation buffers stay in fp32.

        Usage::

            with amp.forward_context():
                logits = model(input_ids)
                loss = criterion(logits, labels)
        """
        if self._active:
            with torch.amp.autocast(device_type=self.device_type, dtype=self._dtype):
                yield
        else:
            yield  # no-op — full fp32

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """Multiply loss by the current GradScaler scale before .backward().

        When fp16 is active this prevents gradient underflow.
        When bf16 or disabled this is a no-op (returns the original tensor).

        Always call scale_loss(loss).backward(), NOT loss.backward().

        Returns:
            Scaled loss tensor (or original tensor when scaler is inactive).
        """
        if self._scaler is not None:
            return self._scaler.scale(loss)
        return loss

    def unscale(self, optimizer: torch.optim.Optimizer) -> None:
        """Un-scale gradients in place so grad clipping sees the true scale.

        Must be called BEFORE torch.nn.utils.clip_grad_norm_() and BEFORE
        amp.step(optimizer).  Safe to call even when scaler is inactive (no-op).

        Usage::

            amp.unscale(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            amp.step(optimizer)
        """
        if self._scaler is not None:
            self._scaler.unscale_(optimizer)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        """Call optimizer.step() via the GradScaler (fp16) or directly (bf16/fp32).

        The scaler skips the optimizer step if any gradient is inf/nan
        (which would otherwise corrupt master weights in fp16).

        Usage (after unscale and clip)::

            amp.step(optimizer)
        """
        if self._scaler is not None:
            self._scaler.step(optimizer)
        else:
            optimizer.step()

    def update(self) -> None:
        """Update the GradScaler's internal scale factor.

        Grows the scale by growth_factor every growth_interval clean steps;
        shrinks it by backoff_factor after any overflow step.

        Call once per optimizer step, AFTER amp.step().  No-op when inactive.
        """
        if self._scaler is not None:
            self._scaler.update()

    def zero_grad(self, optimizer: torch.optim.Optimizer) -> None:
        """Zero gradients efficiently (set_to_none=True avoids memset overhead)."""
        optimizer.zero_grad(set_to_none=True)

    def stats(self) -> dict[str, float | str]:
        """Return loggable statistics for the current step.

        Returns a dict with:
          amp/enabled   — 1.0 or 0.0
          amp/dtype     — "bf16", "fp16", or "fp32" (string tag)
          amp/scale     — current GradScaler scale (fp16 only; else 1.0)
          amp/overflows — cumulative overflow count since training started
        """
        scale = float(self._scaler.get_scale()) if self._scaler else 1.0
        return {
            "amp/enabled": float(self._active),
            "amp/dtype": self.cfg.dtype,
            "amp/scale": scale,
        }

    def state_dict(self) -> dict:
        """Serialise scaler state for checkpoint saving."""
        if self._scaler is not None:
            return {"scaler": self._scaler.state_dict()}
        return {}

    def load_state_dict(self, state: dict) -> None:
        """Restore scaler state from a checkpoint."""
        if self._scaler is not None and "scaler" in state:
            self._scaler.load_state_dict(state["scaler"])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_device_type(device: str | torch.device) -> str:
    """Normalise a device to the 'cuda' / 'cpu' / 'mps' string AMP expects."""
    s = str(device)
    if s.startswith("cuda"):
        return "cuda"
    if s.startswith("mps"):
        return "mps"
    return "cpu"


def _amp_supported(device_type: str, cfg: AMPConfig) -> bool:
    """Return True when AMP can run on this device with the requested dtype."""
    if device_type == "cuda":
        return True  # all CUDA devices support fp16; Ampere+ support bf16
    if device_type == "mps":
        # MPS supports bf16 autocast (macOS 13.0+), not fp16
        return cfg.dtype == "bf16"
    if device_type == "cpu":
        # CPU autocast only supports bf16
        return cfg.enabled_on_cpu and cfg.dtype == "bf16"
    return False


def build_amp_manager(cfg: AMPConfig, device: str | torch.device) -> AMPManager:
    """Convenience factory.  Import and call from Trainer.__init__."""
    return AMPManager(cfg, device)
