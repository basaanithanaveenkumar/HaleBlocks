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

from hale_core.config.sections.quant.export_config import ExportConfig
from hale_core.config.sections.quant.quant_config import QuantConfig

__all__ = ["ExportConfig", "QuantConfig"]
