"""``ExportConfig``: Controls model export to ONNX, TensorRT and MLX."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from hale_core.config.sections.common import StrictModel


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
