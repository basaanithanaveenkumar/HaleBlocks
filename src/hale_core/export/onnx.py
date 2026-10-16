"""
ONNX Export
===========

What is ONNX?
-------------
Open Neural Network Exchange (ONNX) is an open graph format for ML models.
A .onnx file contains:
  - A protobuf-serialised computation graph (nodes = ops, edges = tensors)
  - Opset version: the set of ops available (higher = more modern ops)
  - Named inputs/outputs with shape signatures

Once exported, ONNX models can be run on:
  ONNX Runtime  → CPU/CUDA on Windows, Linux, macOS (fastest cross-platform)
  TensorRT      → NVIDIA GPU inference (via ONNX parser)
  CoreML        → Apple devices (via coremltools converter)
  OpenVINO      → Intel CPUs and Neural Compute Sticks
  RKNN          → Rockchip NPUs (edge devices)

How torch.onnx.export works
---------------------------
PyTorch traces the model by running a dummy forward pass and recording all
operations into a TorchScript graph.  It then maps each TorchScript op to the
corresponding ONNX operator(s) from the opset.

Dynamic axes
------------
By default, ONNX exports fixed shapes.  With dynamic_axes, you specify which
dimensions can vary at runtime:

    dynamic_axes = {
        "input_ids":  {0: "batch_size", 1: "seq_len"},
        "logits":     {0: "batch_size", 1: "seq_len"},
    }

The exported model accepts any value for those dimensions during inference.

onnx-simplifier (onnxsim)
--------------------------
torch.onnx.export often produces redundant nodes (shape operations, identity ops,
unnecessary casts).  onnxsim performs constant folding and dead-node elimination,
making the graph smaller and more readable.  It also helps TensorRT parse the
graph correctly.

    pip install onnxsim

Export output
-------------
  <output_dir>/model.onnx            — the main export
  <output_dir>/model_simplified.onnx — optional simplified version
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from loguru import logger

if TYPE_CHECKING:
    from hale_core.config.sections.quant import ExportConfig


def export_onnx(
    model: nn.Module,
    sample_input: dict[str, torch.Tensor] | torch.Tensor,
    cfg: ExportConfig,
    *,
    output_dir: str | Path | None = None,
    model_name: str = "model",
) -> str:
    """Export model to ONNX format.

    Args:
        model:        The nn.Module to export.  Should be in eval() mode and on
                      CPU (ONNX export is CPU-based).
        sample_input: Dummy input matching the shape used at inference time.
                      Can be a single tensor or a dict of named tensors.
        cfg:          ExportConfig section from TrainConfig.
        output_dir:   Override the output directory from cfg.
        model_name:   Base filename (without extension).

    Returns:
        Absolute path to the saved .onnx file.

    Raises:
        RuntimeError if torch.onnx.export fails.
    """
    out_dir = Path(output_dir or cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = str(out_dir / f"{model_name}.onnx")

    # ONNX export requires the model on CPU
    model_cpu = model.cpu().eval()

    # Normalise sample_input to a tuple (torch.onnx.export arg)
    input_tuple, input_names, output_names, dynamic_axes = _prepare_onnx_args(sample_input, cfg)

    logger.info(
        "Exporting ONNX: opset={}, dynamic_axes={}, path={}",
        cfg.onnx_opset,
        cfg.onnx_dynamic_axes,
        onnx_path,
    )

    with torch.no_grad():
        torch.onnx.export(
            model_cpu,
            input_tuple,
            onnx_path,
            opset_version=cfg.onnx_opset,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes if cfg.onnx_dynamic_axes else None,
            export_params=True,  # embed all weights in the .onnx file
            verbose=False,
        )

    logger.info("ONNX export complete: {}", onnx_path)

    # Validate the exported graph
    try:
        import onnx

        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        logger.info(
            "ONNX graph validated (nodes={}, opset={})",
            len(onnx_model.graph.node),
            cfg.onnx_opset,
        )
    except ImportError:
        logger.warning("onnx package not installed; skipping graph validation (pip install onnx)")
    except Exception as e:
        logger.error("ONNX graph validation failed: {}", e)

    # Optional: simplify
    if cfg.onnx_simplify:
        simplified_path = str(out_dir / f"{model_name}_simplified.onnx")
        _simplify(onnx_path, simplified_path)

    return onnx_path


def _prepare_onnx_args(
    sample_input: dict[str, torch.Tensor] | torch.Tensor,
    cfg: ExportConfig,
) -> tuple[tuple, list[str], list[str], dict]:
    """Build the (args, input_names, output_names, dynamic_axes) expected by torch.onnx.export.

    Returns:
        input_tuple:   The positional args tuple to pass as model input.
        input_names:   Names for ONNX graph input nodes.
        output_names:  Names for ONNX graph output nodes.
        dynamic_axes:  Mapping of tensor name → {dim_idx: dim_name}.
    """
    output_names = ["logits"]

    if isinstance(sample_input, torch.Tensor):
        # Single-tensor input (e.g. image models)
        input_names = ["input"]
        input_tuple = (sample_input.cpu(),)
        dynamic_axes = {
            "input": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"},
        }
    else:
        # Dict input — common for LLM/VLM (input_ids, attention_mask, ...)
        input_names = list(sample_input.keys())
        input_tuple = tuple(v.cpu() for v in sample_input.values())
        dynamic_axes = {}
        for name in input_names:
            # Mark batch (dim 0) and seq_len (dim 1) as dynamic
            dynamic_axes[name] = {0: "batch_size", 1: "seq_len"}
        dynamic_axes["logits"] = {0: "batch_size", 1: "seq_len"}

    return input_tuple, input_names, output_names, dynamic_axes


def _simplify(onnx_path: str, output_path: str) -> None:
    """Run onnxsim to fold constants and remove redundant nodes.

    Requires:  pip install onnxsim
    Falls back gracefully if not installed.
    """
    try:
        import onnx
        from onnxsim import simplify

        model = onnx.load(onnx_path)
        simplified, ok = simplify(model)
        if ok:
            onnx.save(simplified, output_path)
            logger.info("ONNX simplified: {} → {}", onnx_path, output_path)
        else:
            logger.warning("onnxsim could not simplify; keeping original")
    except ImportError:
        logger.debug("onnxsim not installed; skipping simplification (pip install onnxsim)")
    except Exception as e:
        logger.warning("ONNX simplification failed: {}", e)
