"""
QuantConfig — Quantization-Aware Training (QAT) configuration.
ExportConfig — Export targets configuration (ONNX / TensorRT / MLX).

──────────────────────────────────────────────────────────────────────────────
What is QAT?
────────────────────────────────────────────────────────────────────────────
Post-Training Quantization (PTQ) converts weights and activations to int8 after
training is complete.  It is fast but can lose accuracy on sensitive models.

Quantization-Aware Training (QAT) inserts "fake quantization" nodes during
training: weights and activations are rounded to int8 precision at every forward
pass, but the backward pass uses straight-through estimators (STE) so gradients
still flow.  The model learns to be robust to quantization noise, recovering
most or all of the accuracy lost by PTQ.

How PyTorch QAT works
─────────────────────
1. prepare_qat(model, config)  — inserts FakeQuantize observers on activations
                                  and weights at every quantizable layer
2. Train for qat_steps         — the model sees quantized activations on every
                                  forward pass; gradients flow through STE
3. torch.quantization.convert(model) — replaces FakeQuantize with real int8 ops;
                                        model now runs in int8 on CPU/CUDA

Supported backends
──────────────────
"fbgemm"  → Intel x86/x64 CPUs.  Uses vectorised int8 GEMM.
"qnnpack" → ARM CPUs (Android, Raspberry Pi, Apple Silicon CPU).
"x86"     → Modern alias for fbgemm on x86 with additional optimisations.

──────────────────────────────────────────────────────────────────────────────
Export targets
────────────────────────────────────────────────────────────────────────────
ONNX
  Open Neural Network Exchange — cross-platform graph format.
  torch.onnx.export traces or scripts the model into a protobuf file.
  Downstream runtimes: ONNX Runtime (CPU/CUDA), TensorRT, CoreML, OpenVINO.

TensorRT
  NVIDIA's inference runtime.  Takes an ONNX file and compiles it to a highly
  optimised engine for a specific GPU.  Supports:
  - FP32 / FP16 / INT8 precision
  - Layer fusion, kernel auto-tuning, dynamic shapes
  Requires: tensorrt Python package and a CUDA GPU at export time.

MLX
  Apple's array framework for Apple Silicon (M1/M2/M3).  Runs on the unified
  GPU/CPU and supports weight-only quantisation (4-bit/8-bit).
  Exporting to MLX means saving weights in safetensors format and generating
  an mlx_lm-compatible config.json.
  Requires: mlx Python package (macOS only).

──────────────────────────────────────────────────────────────────────────────

QuantConfig fields
──────────────────
enabled           : master switch for QAT
backend           : quantization backend (fbgemm / qnnpack / x86)
per_channel       : per-channel weight quantization (more accurate than per-tensor)
calibration_batches: how many batches to run through the model to calibrate
                     observer statistics before converting to int8
qat_start_step    : global training step to insert fake quant nodes; train FP32
                    first so the model is partially converged before QAT begins
symmetric         : symmetric vs asymmetric quantization (symmetric is simpler,
                    asymmetric is more accurate for ReLU-heavy models)
bits              : quantization precision (8 for int8, 4 for int4 — int4 only
                    supported in some backends)
convert_at_end    : call torch.quantization.convert after training completes

ExportConfig fields
────────────────────
enabled           : master switch for any export
targets           : list of export targets to produce
output_dir        : directory where all exported files are written
onnx_opset        : ONNX opset version (17 supports most recent torch ops)
onnx_dynamic_axes : enable dynamic batch / sequence dimensions in ONNX graph
trt_fp16          : compile TensorRT engine in FP16 (needs trt_fp16-capable GPU)
trt_workspace_mb  : TensorRT engine builder workspace in MB (larger = more fused)
trt_calib_batches : int8 calibration batches for TensorRT INT8 mode
mlx_quantize_bits : weight-only quantization bits for MLX export (4 or 8)
mlx_group_size    : quantization group size for MLX (smaller = more accurate)
sample_seq_len    : sequence length used to build the dummy export input
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

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


class ExportConfig(StrictModel):
    """Controls model export to ONNX, TensorRT and MLX."""

    # ── Master switch ─────────────────────────────────────────────────────────
    enabled: bool = False
    """Set True to export after training completes (or on demand)."""

    # ── Targets ───────────────────────────────────────────────────────────────
    targets: list[Literal["onnx", "tensorrt", "mlx"]] = Field(default_factory=list)
    """Which backends to export to.  Example: ["onnx", "tensorrt"].
    Each target writes its artefacts to output_dir."""

    output_dir: str = "exports"
    """Directory where all exported artefacts are saved.  Created if absent."""

    # ── Dummy input shape ─────────────────────────────────────────────────────
    sample_batch_size: int = 1
    """Batch size for the dummy input used during tracing / export."""

    sample_seq_len: int = 128
    """Sequence length for the dummy input.  Should match the typical inference
    length, not the training max_length, to get best-case engine shapes."""

    # ── ONNX ──────────────────────────────────────────────────────────────────
    onnx_opset: int = 17
    """ONNX opset version.  17 supports flash-attention patterns and most recent
    torch operators; use 14 for older runtimes."""

    onnx_dynamic_axes: bool = True
    """Export ONNX with dynamic batch and sequence dimensions so the same file
    can be used for inference at different shapes."""

    onnx_simplify: bool = True
    """Run onnx-simplifier (onnxsim) after export to fold constants, remove
    redundant nodes and make the graph easier to inspect.
    Requires: pip install onnxsim"""

    # ── TensorRT ──────────────────────────────────────────────────────────────
    trt_fp16: bool = True
    """Build TensorRT engine in FP16 precision.  Requires a GPU with fp16 Tensor
    Core support (Volta+).  Gives ~2× throughput vs FP32."""

    trt_int8: bool = False
    """Build TensorRT engine in INT8 precision.  Requires calibration data.
    Gives ~4× throughput vs FP32 on Tensor Core hardware."""

    trt_workspace_mb: int = 4096
    """Builder workspace in MB.  Larger workspace allows TRT to explore more
    kernel fusion strategies.  Set lower on GPUs with <8 GB VRAM."""

    trt_calib_batches: int = 50
    """INT8 calibration batches for TensorRT.  More = more accurate scale
    statistics.  Only used when trt_int8=True."""

    # ── MLX (Apple Silicon) ───────────────────────────────────────────────────
    mlx_quantize_bits: Literal[4, 8, 16] = 4
    """Weight-only quantization bit-width for MLX export.
    4-bit  → most compression, small accuracy loss with good group_size.
    8-bit  → near FP16 quality; fast on Apple Neural Engine.
    16-bit → no quantization; pure bf16 weights."""

    mlx_group_size: int = 64
    """Quantization group size for MLX 4/8-bit.  Smaller group = more accurate
    but larger metadata overhead.  32 or 64 is the typical sweet-spot."""
