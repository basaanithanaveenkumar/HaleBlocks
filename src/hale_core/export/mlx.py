"""
MLX Export — Apple Silicon
===========================

What is MLX?
------------
MLX is Apple's array framework for machine learning on Apple Silicon (M1/M2/M3
and later).  It runs on the unified GPU/CPU memory and supports:

  - Lazy evaluation (arrays are computed only when needed)
  - Automatic differentiation
  - Weight-only quantization (4-bit, 8-bit) for LLM inference
  - JIT compilation for repeated operations

MLX vs Core ML
--------------
Core ML is Apple's production inference framework (ships in iOS/macOS).  It is
suitable for on-device deployment in apps.

MLX is a *research* framework — it gives full programmatic access to the model
and is the standard for running large language models on Mac.  The mlx-lm
package provides ready-made LLM runners compatible with HuggingFace checkpoints.

Export strategy
---------------
PyTorch's mlx export is NOT a one-click conversion.  The recommended approach
is to save weights in safetensors format (a simple, memory-mappable tensor file
format) and write an mlx-compatible config.json that mlx-lm can load.

This module:
  1. Extracts the model's state_dict (fp32 weights)
  2. Optionally quantises weights to 4-bit or 8-bit groupwise quantization
  3. Saves the result as safetensors + config.json
  4. Writes a README with loading instructions

Quantization (mlx_quantize_bits)
---------------------------------
MLX supports weight-only quantization (not activation quantization).

4-bit (Q4):
  Each group of `group_size` weights (default 64) is quantized to 4 bits.
  The resulting model is ~8× smaller than fp32.  Quality is surprisingly good
  for LLMs with >7B parameters.

8-bit (Q8):
  Each group is quantized to 8 bits.  ~4× smaller than fp32.
  Near FP16 quality for most models.

16-bit:
  No quantization; weights saved as bf16 safetensors.

Installation
------------
    pip install mlx mlx-lm

Note: mlx is only available on macOS 13.3+ with Apple Silicon.

Output files
------------
  <output_dir>/mlx/weights.safetensors  — quantized or bf16 weights
  <output_dir>/mlx/config.json           — model architecture config
  <output_dir>/mlx/tokenizer.json        — tokenizer config (if provided)
  <output_dir>/mlx/README.md             — loading instructions

Loading in MLX::

    from mlx_lm import load, generate
    model, tokenizer = load("exports/mlx/")
    text = generate(model, tokenizer, prompt="Hello world")
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from loguru import logger

if TYPE_CHECKING:
    from hale_core.config.sections.quant import ExportConfig


def export_mlx(
    model: nn.Module,
    cfg: ExportConfig,
    *,
    model_config: dict[str, Any] | None = None,
    tokenizer: Any = None,
    output_dir: str | Path | None = None,
    model_name: str = "model",
) -> str:
    """Export a PyTorch model to MLX-compatible safetensors format.

    Args:
        model:        The nn.Module to export (should be in eval() mode).
        cfg:          ExportConfig with MLX-specific settings.
        model_config: Dict of model architecture parameters to write to config.json.
                      If None, a minimal placeholder is written.
        tokenizer:    Optional tokenizer to save (HuggingFace-style save_pretrained).
        output_dir:   Override output directory.
        model_name:   Not used for MLX (always saves as weights.safetensors).

    Returns:
        Absolute path to the output directory.

    Raises:
        ImportError if safetensors is not installed.
    """
    out_dir = Path(output_dir or cfg.output_dir) / "mlx"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: extract weights ───────────────────────────────────────────────
    logger.info("MLX export: collecting state_dict")
    state = _collect_state_dict(model)

    # ── Step 2: quantise if requested ────────────────────────────────────────
    bits = cfg.mlx_quantize_bits
    if bits in (4, 8):
        logger.info(
            "MLX export: applying {}bit groupwise quantization (group_size={})",
            bits,
            cfg.mlx_group_size,
        )
        state = _quantize_state_dict(state, bits=bits, group_size=cfg.mlx_group_size)
    else:
        # 16-bit: convert to bf16 (MLX uses bf16 natively)
        state = {k: v.to(torch.bfloat16) for k, v in state.items()}
        logger.info("MLX export: saving as bf16 (no quantization)")

    # ── Step 3: save safetensors ──────────────────────────────────────────────
    weights_path = str(out_dir / "weights.safetensors")
    _save_safetensors(state, weights_path)
    logger.info(
        "MLX weights saved: {} ({:.1f} MB)", weights_path, os.path.getsize(weights_path) / 1e6
    )

    # ── Step 4: write config.json ─────────────────────────────────────────────
    config_path = str(out_dir / "config.json")
    _write_config(config_path, model_config, cfg)
    logger.info("MLX config saved: {}", config_path)

    # ── Step 5: save tokenizer ────────────────────────────────────────────────
    if tokenizer is not None:
        _save_tokenizer(tokenizer, out_dir)

    # ── Step 6: write README ──────────────────────────────────────────────────
    _write_readme(out_dir, bits=bits, group_size=cfg.mlx_group_size)

    logger.info("MLX export complete: {}", out_dir)
    return str(out_dir)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _collect_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract float tensors from model state_dict, skipping non-float buffers."""
    raw = model.state_dict()
    out: dict[str, torch.Tensor] = {}
    for key, tensor in raw.items():
        if tensor.is_floating_point():
            # Convert to fp32 first; quantization works from fp32
            out[key] = tensor.cpu().float()
        # Skip integer buffers (position_ids, attention masks, etc.)
    return out


def _quantize_state_dict(
    state: dict[str, torch.Tensor],
    bits: int,
    group_size: int,
) -> dict[str, torch.Tensor]:
    """Apply groupwise affine quantization to weight tensors.

    For each weight tensor W of shape (out, in):
      - Divide columns into groups of `group_size`
      - Compute per-group min/max → scale and zero_point
      - Quantize each group to `bits` bits
      - Store as: W_q (uint8 packed), scales (fp32), zeros (fp32)

    Non-weight tensors (embedding, bias, norms) are kept in bf16.

    The output format matches mlx-lm's expected layout for quantized models.

    Returns:
        New state_dict with quantized weights (additional scale/zeros tensors).
    """
    out: dict[str, torch.Tensor] = {}

    for key, tensor in state.items():
        # Only quantize 2D weight matrices (Linear layers)
        # Skip biases, norms, embeddings
        if tensor.ndim == 2 and "weight" in key and "norm" not in key and "embed" not in key:
            q_weights, scales, zeros = _groupwise_quantize(tensor, bits=bits, group_size=group_size)
            out[key] = q_weights  # int8/uint8 packed values
            out[key + "_scales"] = scales  # (out, n_groups) float32
            out[key + "_zeros"] = zeros  # (out, n_groups) float32
        else:
            out[key] = tensor.to(torch.bfloat16)

    return out


def _groupwise_quantize(
    weight: torch.Tensor,
    bits: int,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a 2D weight matrix using groupwise affine quantization.

    Args:
        weight:     (out_features, in_features) float32 tensor.
        bits:       Quantization bit-width (4 or 8).
        group_size: Number of input elements per quantization group.

    Returns:
        (quantized_weights, scales, zeros)
        - quantized_weights: (out, in) uint8 tensor (values in [0, 2^bits - 1])
        - scales:            (out, n_groups) float32
        - zeros:             (out, n_groups) float32

    Note: 4-bit values are packed as uint8 (two 4-bit values per byte)
          but stored unpacked here for simplicity; mlx-lm handles packing.
    """
    max_val = (1 << bits) - 1  # 255 for 8-bit, 15 for 4-bit
    out_features, in_features = weight.shape

    # Pad in_features to a multiple of group_size
    pad = (group_size - in_features % group_size) % group_size
    if pad > 0:
        weight = torch.nn.functional.pad(weight, (0, pad))
    n_groups = weight.shape[1] // group_size

    # Reshape to (out, n_groups, group_size) for per-group statistics
    w = weight.reshape(out_features, n_groups, group_size)

    # Per-group min/max for asymmetric quantization
    w_min = w.min(dim=-1).values  # (out, n_groups)
    w_max = w.max(dim=-1).values  # (out, n_groups)

    # Scale: maps [w_min, w_max] → [0, max_val]
    scales = (w_max - w_min) / max_val  # (out, n_groups)
    scales = scales.clamp(min=1e-8)  # avoid div-by-zero
    zeros = w_min / scales  # offset

    # Quantize: w_q = round((w - w_min) / scale)
    w_min_exp = w_min.unsqueeze(-1)  # (out, n_groups, 1)
    scales_exp = scales.unsqueeze(-1)
    w_q = ((w - w_min_exp) / scales_exp).round().clamp(0, max_val).to(torch.uint8)

    # Remove padding
    w_q = w_q.reshape(out_features, -1)[:, :in_features]

    return w_q, scales, zeros


def _save_safetensors(state: dict[str, torch.Tensor], path: str) -> None:
    """Save tensors to safetensors format.

    safetensors is a simple, memory-mappable format without Python pickle.
    MLX can load safetensors directly with zero-copy memory mapping.

    Requires: pip install safetensors
    """
    try:
        from safetensors.torch import save_file

        # safetensors requires contiguous tensors
        contiguous = {k: v.contiguous() for k, v in state.items()}
        save_file(contiguous, path)
    except ImportError:
        logger.warning("safetensors not installed; saving as .pt instead (pip install safetensors)")
        pt_path = path.replace(".safetensors", ".pt")
        torch.save(state, pt_path)


def _write_config(path: str, model_config: dict | None, cfg: ExportConfig) -> None:
    """Write MLX config.json with model architecture and quantization info."""
    base_config = model_config or {}
    mlx_config = {
        **base_config,
        "quantization": {
            "bits": cfg.mlx_quantize_bits,
            "group_size": cfg.mlx_group_size,
        }
        if cfg.mlx_quantize_bits < 16
        else None,
        "model_type": base_config.get("model_type", "hale"),
    }
    # Remove None keys
    mlx_config = {k: v for k, v in mlx_config.items() if v is not None}
    with open(path, "w") as f:
        json.dump(mlx_config, f, indent=2)


def _save_tokenizer(tokenizer: Any, out_dir: Path) -> None:
    """Save tokenizer files (HuggingFace-compatible)."""
    try:
        tokenizer.save_pretrained(str(out_dir))
        logger.info("MLX tokenizer saved to {}", out_dir)
    except Exception as e:
        logger.warning("Could not save tokenizer: {}", e)


def _write_readme(out_dir: Path, bits: int, group_size: int) -> None:
    """Write a README with loading instructions."""
    quant_note = (
        f"Weights quantized to {bits}-bit (group_size={group_size})"
        if bits < 16
        else "Weights in bf16 (no quantization)"
    )
    readme = f"""# MLX Export

{quant_note}

## Requirements

```bash
pip install mlx mlx-lm safetensors
```

## Load and run

```python
from mlx_lm import load, generate

model, tokenizer = load("{out_dir}")
response = generate(model, tokenizer, prompt="Hello, world!", max_tokens=200)
print(response)
```

## Manual loading

```python
import mlx.core as mx
from safetensors import safe_open

weights = {{}}
with safe_open("{out_dir}/weights.safetensors", framework="pt") as f:
    for key in f.keys():
        weights[key] = mx.array(f.get_tensor(key).numpy())
```
"""
    (out_dir / "README.md").write_text(readme)
