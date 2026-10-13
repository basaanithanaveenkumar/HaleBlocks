"""
TensorRT Export
===============

What is TensorRT?
-----------------
NVIDIA TensorRT (TRT) is a high-performance deep learning inference optimizer
and runtime for NVIDIA GPUs.  Given a trained model (as ONNX), TensorRT:

  1. Parses the ONNX graph
  2. Fuses adjacent layers (Conv + BN + ReLU → one kernel)
  3. Selects the fastest CUDA kernel for each layer on the target GPU
  4. Compiles everything into a single serialised "engine" file (.trt / .engine)

The resulting engine runs only on the specific GPU architecture it was compiled
for.  You cannot move an H100 engine to a T4.

Precision modes
---------------
FP32   → default; numerically identical to PyTorch; slowest
FP16   → 2× throughput on Tensor Core GPUs; small accuracy loss; recommended
INT8   → 4× throughput; requires calibration data; significant accuracy loss
         unless QAT was used

Dynamic shapes
--------------
TRT engines can accept variable input shapes within a declared range:
  - min_shape: smallest valid input
  - opt_shape: the "typical" input (used for best kernel selection)
  - max_shape: largest valid input

For sequence models, typically:
  min_shape = (1, 1)       # batch=1, seq_len=1
  opt_shape = (8, 128)     # typical inference batch and length
  max_shape = (32, 2048)   # maximum supported

Installation
------------
TensorRT is NOT a pip package.  Install it from:
  https://developer.nvidia.com/tensorrt

After installing, the Python bindings are available as:
  import tensorrt as trt

The tensorrt package only works on machines with a matching NVIDIA GPU and
CUDA toolkit.  Export is always done on the target GPU.

Output
------
  <output_dir>/model.trt        — serialised engine (binary)

Usage at inference time:
    import tensorrt as trt
    from hale_core.export.tensorrt import TRTInferenceSession

    session = TRTInferenceSession("exports/model.trt")
    output = session.run(input_ids=input_ids, attention_mask=mask)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from loguru import logger

if TYPE_CHECKING:
    from hale_core.config.sections.quant import ExportConfig


def export_tensorrt(
    onnx_path: str | Path,
    cfg: ExportConfig,
    *,
    output_dir: str | Path | None = None,
    model_name: str = "model",
    input_shapes: dict[str, tuple[tuple, tuple, tuple]] | None = None,
) -> str | None:
    """Compile an ONNX model into a TensorRT engine.

    Args:
        onnx_path:     Path to the .onnx file (produced by export_onnx).
        cfg:           ExportConfig with TRT-specific settings.
        output_dir:    Override output directory.
        model_name:    Base filename (without extension).
        input_shapes:  Per-input shape profiles for dynamic shapes.
                       Format: {"input_name": (min, opt, max)} where each
                       shape is a tuple of ints, e.g. (1, 1) for (batch, seq).
                       If None, static shapes from the ONNX file are used.

    Returns:
        Path to the .trt engine file, or None if TensorRT is not installed.

    Notes:
        - Requires tensorrt Python package and a CUDA GPU.
        - Engine is GPU-architecture-specific; not portable across GPU families.
        - FP16 mode requires a GPU with fp16 Tensor Core support (Volta+).
        - INT8 mode requires calibration data (not yet implemented here; use
          TRT's Python API directly for INT8 calibration).
    """
    try:
        import tensorrt as trt
    except ImportError:
        logger.warning(
            "tensorrt not installed; skipping TRT export. "
            "Install from: https://developer.nvidia.com/tensorrt"
        )
        return None

    out_dir = Path(output_dir or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    engine_path = str(out_dir / f"{model_name}.trt")

    # ── Build configuration ───────────────────────────────────────────────────
    trt_logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(trt_logger)

    # INetworkDefinition: the parsed model graph
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, trt_logger)

    # Parse ONNX graph
    with open(onnx_path, "rb") as f:
        raw = f.read()
    if not parser.parse(raw):
        for i in range(parser.num_errors):
            logger.error("TRT ONNX parse error: {}", parser.get_error(i))
        raise RuntimeError(f"TensorRT failed to parse ONNX: {onnx_path}")

    logger.info(
        "TRT: parsed ONNX ({} inputs, {} layers)",
        network.num_inputs,
        network.num_layers,
    )

    # ── Build config ──────────────────────────────────────────────────────────
    build_config = builder.create_builder_config()
    build_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, cfg.trt_workspace_mb * 1024 * 1024
    )

    if cfg.trt_fp16 and builder.platform_has_fast_fp16:
        build_config.set_flag(trt.BuilderFlag.FP16)
        logger.info("TRT: FP16 mode enabled")
    elif cfg.trt_fp16:
        logger.warning("TRT: FP16 requested but GPU does not support fp16 Tensor Cores")

    # Dynamic shapes profile
    if input_shapes:
        profile = builder.create_optimization_profile()
        for input_name, (min_s, opt_s, max_s) in input_shapes.items():
            profile.set_shape(input_name, min_s, opt_s, max_s)
        build_config.add_optimization_profile(profile)
        logger.info("TRT: dynamic shapes configured for {} inputs", len(input_shapes))

    # ── Compile engine ────────────────────────────────────────────────────────
    logger.info("TRT: building engine (this may take several minutes)…")
    serialised = builder.build_serialized_network(network, build_config)
    if serialised is None:
        raise RuntimeError("TensorRT engine build failed (check GPU memory and opset support)")

    with open(engine_path, "wb") as f:
        f.write(serialised)

    logger.info("TRT engine saved: {} ({:.1f} MB)", engine_path, os.path.getsize(engine_path) / 1e6)
    return engine_path


# ── Inference helper ──────────────────────────────────────────────────────────

class TRTInferenceSession:
    """Load and run a serialised TensorRT engine.

    This is a convenience wrapper for the common case of a single-stream
    inference loop.  For production use, consider using NVIDIA's Triton
    Inference Server or TensorRT's C++ API directly.

    Example::

        session = TRTInferenceSession("exports/model.trt")
        logits = session.run(input_ids=ids_tensor, attention_mask=mask_tensor)
    """

    def __init__(self, engine_path: str | Path) -> None:
        try:
            import tensorrt as trt
        except ImportError:
            raise RuntimeError("tensorrt package required for TRTInferenceSession")

        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)

        with open(engine_path, "rb") as f:
            self._engine = runtime.deserialize_cuda_engine(f.read())

        self._context = self._engine.create_execution_context()
        self._stream = torch.cuda.Stream()

    def run(self, **kwargs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Execute the engine with the given named input tensors.

        Args:
            **kwargs: Named input tensors (names must match ONNX export input names).

        Returns:
            Dict of output name → output tensor (on GPU).
        """
        import tensorrt as trt

        # Bind inputs
        bindings: list[int] = []
        output_buffers: dict[str, torch.Tensor] = {}

        for i in range(self._engine.num_io_tensors):
            name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(name)
            dtype = _trt_to_torch_dtype(self._engine.get_tensor_dtype(name))

            if mode == trt.TensorIOMode.INPUT:
                tensor = kwargs[name].cuda().contiguous()
                self._context.set_input_shape(name, tuple(tensor.shape))
            else:
                shape = tuple(self._context.get_tensor_shape(name))
                tensor = torch.empty(shape, dtype=dtype, device="cuda")
                output_buffers[name] = tensor

            bindings.append(tensor.data_ptr())

        with torch.cuda.stream(self._stream):
            self._context.execute_async_v3(self._stream.cuda_stream)
        self._stream.synchronize()

        return output_buffers


def _trt_to_torch_dtype(trt_dtype) -> torch.dtype:
    """Map TensorRT dtype enum to torch.dtype."""
    try:
        import tensorrt as trt
        mapping = {
            trt.float32: torch.float32,
            trt.float16: torch.float16,
            trt.int32:   torch.int32,
            trt.int8:    torch.int8,
        }
        return mapping.get(trt_dtype, torch.float32)
    except Exception:
        return torch.float32
