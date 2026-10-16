"""
AMPConfig — configuration for Automatic Mixed Precision (AMP) training.

Mixed precision training runs the forward pass and loss in a lower-precision
floating-point type (float16 or bfloat16) while keeping the master weights in
float32. This gives:
  - ~2× memory reduction for activations, gradients and optimizer state
  - ~2-3× throughput gain on Tensor Core hardware (A100, H100, RTX-series)
  - Numerically equivalent results for most models when bf16 or the GradScaler
    are used correctly

How AMP works in PyTorch
------------------------
torch.amp.autocast wraps the forward pass.  Inside the context, eligible ops
(matmul, conv, attention) run in fp16/bf16; others stay in fp32.  The result is
still accumulated in fp32 before the backward pass.

When using fp16 (NOT bf16), gradients can underflow to zero because fp16's
dynamic range is only ≈ 1e-4 → 6e4.  The GradScaler multiplies the loss by a
large scale factor before .backward(), then divides the gradients back before
the optimizer step.  The scale is automatically adjusted: it grows when no inf/
nan appears, and shrinks when one does.

bf16 has the same exponent range as fp32 (just fewer mantissa bits), so it
rarely needs a scaler.  On A100/H100, bf16 is usually the better choice.

Field-by-field
--------------
enabled       : master switch; False = full fp32 (no overhead)
dtype         : "fp16" → GradScaler required; "bf16" → scaler optional but safe
                to leave on; "fp32" → AMP is effectively a no-op
init_scale    : initial loss scale for GradScaler (2^16 = 65536 is PyTorch default)
growth_factor : multiply scale by this after growth_interval consecutive valid steps
backoff_factor: multiply scale by this after an inf/nan is detected (< 1.0)
growth_interval: steps between scale growth attempts
enabled_on_cpu: whether to use autocast on CPU (only bf16 is supported on CPU)
"""

from __future__ import annotations

from typing import Literal

from hale_core.config.sections.common import StrictModel


class AMPConfig(StrictModel):
    """Controls automatic mixed-precision training."""

    # ── Master switch ─────────────────────────────────────────────────────────
    enabled: bool = False
    """Set True to activate AMP.  Safe to enable on any CUDA GPU; on CPU only
    bf16 is supported (set enabled_on_cpu=True explicitly)."""

    # ── Precision dtype ───────────────────────────────────────────────────────
    dtype: Literal["fp16", "bf16", "fp32"] = "bf16"
    """
    bf16  — best for Ampere+ (A100, H100, RTX 3090+).  Wide dynamic range,
            no scaler needed, numerically stable.
    fp16  — best for Volta/Turing (V100, T4, RTX 20/30-series).  Needs scaler.
    fp32  — disables casting; autocast becomes a no-op (useful for debugging).
    """

    # ── GradScaler (fp16 only) ────────────────────────────────────────────────
    init_scale: float = 65536.0
    """Initial loss scale factor.  2^16 works for most models; lower if you see
    frequent inf/nan at step 0."""

    growth_factor: float = 2.0
    """Scale is multiplied by this value every growth_interval non-overflow steps.
    Higher = more aggressive; use 2.0 (default) for stability."""

    backoff_factor: float = 0.5
    """Scale is multiplied by this after each overflow step.  Must be < 1.0."""

    growth_interval: int = 2000
    """Number of consecutive successful steps before scale grows."""

    # ── CPU AMP ──────────────────────────────────────────────────────────────
    enabled_on_cpu: bool = False
    """Allow autocast on CPU.  PyTorch supports bf16 autocast on CPU; fp16
    on CPU is experimental and may be slow."""
