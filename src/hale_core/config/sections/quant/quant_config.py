"""``QuantConfig``: Quantization-Aware Training settings."""

from __future__ import annotations

from typing import Literal

from hale_core.config.sections.common import StrictModel


class QuantConfig(StrictModel):
    """Quantization-Aware Training settings."""

    # ── Master switch ─────────────────────────────────────────────────────────
    enabled: bool = False
    """Set True to insert fake-quantization nodes and run QAT."""

    # ── Backend ───────────────────────────────────────────────────────────────
    backend: Literal["fbgemm", "qnnpack", "x86"] = "fbgemm"
    """
    fbgemm  — Intel x86 CPUs; vectorised int8 GEMM; best for server inference.
    qnnpack — ARM CPUs; Apple M-series CPU side, Raspberry Pi, Android.
    x86     — Modern alias for fbgemm with AVX-512 and other extras.
    """

    # ── Quantization precision ────────────────────────────────────────────────
    bits: Literal[8, 4] = 8
    """Quantization bit-width.  8 = int8 (widely supported); 4 = int4 (only some
    backends, mostly for weight-only quant in MLX / GPTQ style)."""

    per_channel: bool = True
    """Per-channel weight quantization: each output channel gets its own scale +
    zero-point.  More accurate than per-tensor but slightly more complex.
    Use per_channel=True for Linear layers (default); per_channel=False for Conv1d."""

    symmetric: bool = True
    """Symmetric quantization maps [-max, +max] → int8; zero-point = 0.
    Asymmetric maps [min, max] → int8; non-zero zero-point.
    Symmetric is simpler and preferred for weights; asymmetric is better for
    ReLU activations which are non-negative."""

    # ── Observer calibration ──────────────────────────────────────────────────
    calibration_batches: int = 100
    """Number of forward passes to run with observers active to collect
    min/max / histogram statistics of activations before QAT begins."""

    # ── QAT start step ────────────────────────────────────────────────────────
    qat_start_step: int = 0
    """Insert fake-quant at this global step.  0 = quantize from the start.
    Setting this > 0 lets the model train in FP32 first, which provides a better
    initialisation point for QAT."""

    # ── Post-training conversion ──────────────────────────────────────────────
    convert_at_end: bool = True
    """Call torch.quantization.convert() after training, replacing FakeQuantize
    nodes with real int8 operators ready for inference."""
