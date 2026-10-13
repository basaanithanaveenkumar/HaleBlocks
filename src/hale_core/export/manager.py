"""
ExportManager — unified export orchestrator
============================================

This class is the single entry point for post-training model export.
It reads the ExportConfig and calls the appropriate backend exporter for each
requested target (ONNX, TensorRT, MLX) in the correct order.

Order matters because:
  - TensorRT requires a valid .onnx file; ONNX must run first.
  - MLX reads from the PyTorch state_dict directly; order doesn't matter.

Typical integration in Trainer.fit():

    # after training loop ends:
    if self.cfg.train.export.enabled:
        exporter = ExportManager(
            cfg=self.cfg.train.export,
            model=self.model,
            device=self.device,
            tokenizer=self.tokenizer,
            model_config=self.cfg.model.model_dump(),
        )
        results = exporter.run_all()
        logger.info("export results: {}", results)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from loguru import logger

from hale_core.export.onnx import export_onnx
from hale_core.export.tensorrt import export_tensorrt
from hale_core.export.mlx import export_mlx

if TYPE_CHECKING:
    from hale_core.config.sections.quant import ExportConfig


class ExportManager:
    """Orchestrates export to one or more deployment backends.

    Args:
        cfg:          ExportConfig from TrainConfig.
        model:        Trained nn.Module (should be unwrapped from DDP/FSDP).
        device:       Training device (models are moved to CPU internally as needed).
        tokenizer:    Optional tokenizer (used by MLX export for saving).
        model_config: Dict of model architecture args for config.json (MLX).
        model_name:   Base filename for exported artefacts (default "model").

    Example::

        exporter = ExportManager(
            cfg=run_cfg.train.export,
            model=trainer.model,
            device=trainer.device,
            tokenizer=trainer.tokenizer,
            model_config=run_cfg.model.model_dump(),
        )
        results = exporter.run_all()
        # {"onnx": "exports/model.onnx",
        #  "tensorrt": "exports/model.trt",
        #  "mlx": "exports/mlx/"}
    """

    def __init__(
        self,
        cfg: ExportConfig,
        model: nn.Module,
        device: str | torch.device,
        *,
        tokenizer: Any = None,
        model_config: dict[str, Any] | None = None,
        model_name: str = "model",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.model_name = model_name

        # Unwrap DDP/FSDP/DataParallel wrappers to get the raw module
        self.model = _unwrap_model(model)

    # ── Public API ────────────────────────────────────────────────────────────

    def run_all(
        self,
        sample_input: dict[str, torch.Tensor] | torch.Tensor | None = None,
    ) -> dict[str, str | None]:
        """Export to all configured targets and return a map of target → path.

        Args:
            sample_input: Dummy input for ONNX/TRT tracing.  If None, a dummy
                          input of shape (batch, seq_len) is auto-generated from
                          the export config.

        Returns:
            Dict mapping target name → output path (None if export failed).

        Example::

            results = exporter.run_all(
                sample_input={"input_ids": torch.randint(0, 1000, (1, 128))}
            )
        """
        if not self.cfg.enabled:
            logger.debug("ExportManager: disabled (export.enabled=False)")
            return {}

        if sample_input is None:
            sample_input = self._make_dummy_input()

        results: dict[str, str | None] = {}
        onnx_path: str | None = None

        # ONNX must run first (TRT depends on it)
        targets = self.cfg.targets

        if "onnx" in targets or "tensorrt" in targets:
            try:
                onnx_path = self._export_onnx(sample_input)
                results["onnx"] = onnx_path
            except Exception as e:
                logger.error("ONNX export failed: {}", e)
                results["onnx"] = None

        if "tensorrt" in targets:
            if onnx_path is None:
                logger.error("TRT export skipped: ONNX export failed or not run")
                results["tensorrt"] = None
            else:
                try:
                    results["tensorrt"] = self._export_tensorrt(onnx_path, sample_input)
                except Exception as e:
                    logger.error("TensorRT export failed: {}", e)
                    results["tensorrt"] = None

        if "mlx" in targets:
            try:
                results["mlx"] = self._export_mlx()
            except Exception as e:
                logger.error("MLX export failed: {}", e)
                results["mlx"] = None

        _log_summary(results)
        return results

    # ── Per-target exports ────────────────────────────────────────────────────

    def _export_onnx(
        self, sample_input: dict[str, torch.Tensor] | torch.Tensor
    ) -> str:
        return export_onnx(
            model=self.model,
            sample_input=sample_input,
            cfg=self.cfg,
            model_name=self.model_name,
        )

    def _export_tensorrt(
        self,
        onnx_path: str,
        sample_input: dict[str, torch.Tensor] | torch.Tensor,
    ) -> str | None:
        # Build dynamic shape profiles from sample_input
        input_shapes = _build_trt_profiles(sample_input, self.cfg)
        return export_tensorrt(
            onnx_path=onnx_path,
            cfg=self.cfg,
            model_name=self.model_name,
            input_shapes=input_shapes,
        )

    def _export_mlx(self) -> str:
        return export_mlx(
            model=self.model,
            cfg=self.cfg,
            model_config=self.model_config,
            tokenizer=self.tokenizer,
            model_name=self.model_name,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _make_dummy_input(self) -> dict[str, torch.Tensor]:
        """Build a dummy (batch, seq_len) integer input for text models."""
        B = self.cfg.sample_batch_size
        S = self.cfg.sample_seq_len
        return {
            "input_ids":      torch.zeros(B, S, dtype=torch.long),
            "attention_mask": torch.ones(B, S, dtype=torch.long),
        }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _unwrap_model(model: nn.Module) -> nn.Module:
    """Strip DDP/FSDP/DataParallel wrappers so we get the raw module."""
    while hasattr(model, "module"):
        model = model.module
    return model


def _build_trt_profiles(
    sample_input: dict[str, torch.Tensor] | torch.Tensor,
    cfg: ExportConfig,
) -> dict[str, tuple[tuple, tuple, tuple]]:
    """Build min/opt/max shape profiles for TRT dynamic shapes.

    Uses the sample_input shape as the optimal profile;
    min is half the opt size, max is twice.
    """
    if isinstance(sample_input, torch.Tensor):
        shape = tuple(sample_input.shape)
        return {
            "input": (
                (1, max(1, shape[1] // 2)),    # min
                shape,                          # opt
                (shape[0] * 2, shape[1] * 2),  # max
            )
        }
    profiles: dict[str, tuple[tuple, tuple, tuple]] = {}
    for name, t in sample_input.items():
        B, S = t.shape[0], t.shape[1] if t.ndim > 1 else 1
        profiles[name] = (
            (1, max(1, S // 2)),
            (B, S),
            (B * 2, S * 2),
        )
    return profiles


def _log_summary(results: dict[str, str | None]) -> None:
    """Print a clean summary of export results."""
    logger.info("── Export summary ──────────────────────────────")
    for target, path in results.items():
        status = path if path else "FAILED"
        logger.info("  {:12s} → {}", target, status)
    logger.info("────────────────────────────────────────────────")
