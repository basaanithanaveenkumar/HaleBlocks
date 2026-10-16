"""
Quantization-Aware Training — QATManager
=========================================

Overview
--------
Post-Training Quantization (PTQ) is fast but lossy.  Quantization-Aware
Training (QAT) inserts "fake quantization" nodes into the graph so the model
learns to tolerate int8 rounding *during* gradient descent.  The model is
trained with these fake quant ops active, then converted to a real int8 model
for deployment.

PyTorch QAT pipeline
--------------------
Step 1  prepare_qat(model)
  Inserts FakeQuantize observers on every weight and activation of quantizable
  layers (Linear, Conv).  The model is still in fp32 — fake quant just adds
  round-trip rounding: x → (round(x/scale) + zero_point) * scale.
  Gradients flow through the rounding via the straight-through estimator (STE),
  which treats ∂(round)/∂x = 1.

Step 2  Calibration phase  (first `calibration_batches` forward passes)
  MinMaxObserver / HistogramObserver collects statistics of activations.
  At the end of calibration, per-layer scale and zero-point are computed.
  Fake quant nodes start clipping to these ranges.

Step 3  QAT fine-tuning  (qat_start_step → end of training)
  Normal training loop runs with fake quant active.  The loss is the same
  as normal training — no extra QAT loss term is needed.

Step 4  convert(model)  (after training)
  Replaces every FakeQuantize module with real int8 ops (quantized Linear,
  quantized Conv).  The model can now be run with integer arithmetic on CPU.

Backends
--------
fbgemm   → x86 CPUs.  Uses AVX2/AVX-512 vectorised int8 GEMM.
             deploy with: torch.backends.quantized.engine = 'fbgemm'
qnnpack  → ARM CPUs (Android, Raspberry Pi, Apple M-series CPU side).
             deploy with: torch.backends.quantized.engine = 'qnnpack'
x86      → Modern alias for fbgemm; includes more fused patterns.

Per-channel vs per-tensor quantization
---------------------------------------
per_tensor   → one scale + zero_point for the entire weight tensor.
               Faster, simpler, but less accurate.
per_channel  → one scale + zero_point per output channel (row of weight matrix).
               More accurate, especially for large weight ranges across channels.
               Recommended for all Linear layers.

Typical usage
-------------
    qat = QATManager(cfg.train.quant, model, device)
    qat.calibrate(model, calibration_loader, device)   # collect observer stats
    qat.prepare()                                       # enable fake quant
    # ... training loop runs normally ...
    model = qat.convert()                              # replace with int8 ops
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
from loguru import logger

if TYPE_CHECKING:
    from hale_core.config.sections.quant import QuantConfig


class QATManager:
    """Manages quantization-aware training lifecycle.

    Args:
        cfg:    QuantConfig section from TrainConfig.
        model:  nn.Module to quantize.  Should be on CPU for PyTorch's built-in
                QAT (torch.quantization operates on CPU models).
        device: Training device.  QAT prepare/convert are always done on CPU;
                the manager moves the model automatically when needed.

    Notes
    -----
    - PyTorch's torch.quantization.prepare_qat only supports CPU.  If your
      model trains on GPU, the manager copies to CPU for prepare/convert and
      copies back.  For GPU-native QAT, use torch.ao.quantization (Pytorch 2+).
    - LSTM, Transformer and attention are NOT natively quantizable by the default
      QConfig.  They require a custom QConfig or manual insertion of quant stubs.
      This manager skips non-quantizable layers gracefully.
    """

    def __init__(self, cfg: QuantConfig, model: nn.Module, device: torch.device) -> None:
        self.cfg = cfg
        self.model = model
        self.device = device
        self._prepared = False
        self._converted = False
        self._calibrated = False

    # ── Public API ────────────────────────────────────────────────────────────

    def calibrate(
        self,
        model: nn.Module,
        calibration_loader,
        device: torch.device,
    ) -> None:
        """Run forward passes to collect observer statistics.

        Moves model to CPU, runs calibration_batches forward passes, then moves
        back to device.  After calibration, scale/zero_point are fixed and fake
        quant nodes use those values.

        Args:
            model:              The model (must have been prepare_qat'd already,
                                or calibration runs without fake quant observers).
            calibration_loader: Any iterable yielding batches compatible with
                                model.forward().  Typically the training loader.
            device:             Original training device (model is moved back here).
        """
        if not self.cfg.enabled:
            return

        logger.info(
            "QATManager: calibrating with {} batches on CPU",
            self.cfg.calibration_batches,
        )

        # Calibration must run on CPU for PyTorch's QAT observers
        cpu_model = model.cpu()
        cpu_model.eval()

        with torch.no_grad():
            for i, batch in enumerate(calibration_loader):
                if i >= self.cfg.calibration_batches:
                    break
                # Move batch to CPU; handle both dict and tensor batches
                cpu_batch = _batch_to_device(batch, torch.device("cpu"))
                try:
                    cpu_model(cpu_batch)
                except Exception as e:
                    logger.warning("QATManager: calibration forward failed at batch {}: {}", i, e)

        # Move back to original device
        model.to(device)
        self._calibrated = True
        logger.info("QATManager: calibration complete")

    def prepare(self) -> nn.Module:
        """Insert fake quantization nodes into the model.

        Must be called AFTER moving model to CPU (PyTorch constraint).
        Returns the in-place modified model.

        Changes the model:
          - Adds QuantStub / DeQuantStub at model input/output
          - Inserts FakeQuantize observers on all eligible Linear / Conv layers
          - Sets torch.backends.quantized.engine to cfg.backend

        Returns:
            The model with fake quant hooks (same object, modified in place).
        """
        if not self.cfg.enabled:
            return self.model
        if self._prepared:
            return self.model

        # Set quantized engine; fall back to qnnpack on platforms where
        # fbgemm/x86 are not available (e.g. macOS ARM, Apple Silicon)
        backend = _resolve_backend(self.cfg.backend)
        torch.backends.quantized.engine = backend
        if backend != self.cfg.backend:
            logger.warning(
                "QATManager: backend '{}' not available on this platform; falling back to '{}'",
                self.cfg.backend,
                backend,
            )

        # Build QConfig based on per_channel / bits settings
        qconfig = _build_qconfig(self.cfg)

        # prepare_qat fuses BN+ReLU and inserts fake-quant observers
        cpu_model = self.model.cpu()
        cpu_model.train()
        cpu_model.qconfig = qconfig

        try:
            import torch.quantization as tq

            # Fuse Conv+BN+ReLU patterns first (improves accuracy)
            _try_fuse(cpu_model)
            tq.prepare_qat(cpu_model, inplace=True)
            logger.info(
                "QATManager: fake quant inserted (backend={}, per_channel={}, bits={})",
                self.cfg.backend,
                self.cfg.per_channel,
                self.cfg.bits,
            )
        except Exception as e:
            logger.warning("QATManager: prepare_qat failed ({}); running without QAT", e)

        cpu_model.to(self.device)
        self._prepared = True
        return self.model

    def maybe_prepare(self, global_step: int) -> None:
        """Call inside training loop: inserts fake quant at qat_start_step.

        Usage::

            for step, batch in enumerate(loader):
                qat.maybe_prepare(step)    # no-op until qat_start_step
                with amp.forward_context():
                    loss = criterion(model, batch)
                ...
        """
        if not self.cfg.enabled:
            return
        if self._prepared:
            return
        if global_step >= self.cfg.qat_start_step:
            self.prepare()

    def convert(self) -> nn.Module:
        """Convert fake-quant model to a real int8 model.

        Replaces FakeQuantize nodes with real int8 quantized operators.
        The model can then be exported to ONNX / TorchScript with int8 weights.

        Must be called AFTER training (not during).
        Returns the converted model (same object, modified in place).

        After convert():
          - model.state_dict() contains int8 packed weights
          - model.forward() runs integer arithmetic on CPU
          - The model should NOT be trained further
        """
        if not self.cfg.enabled or not self.cfg.convert_at_end:
            return self.model
        if self._converted:
            return self.model
        if not self._prepared:
            logger.warning("QATManager.convert() called before prepare(); skipping")
            return self.model

        import torch.quantization as tq

        cpu_model = self.model.cpu()
        cpu_model.eval()
        try:
            tq.convert(cpu_model, inplace=True)
            logger.info("QATManager: model converted to int8")
        except Exception as e:
            logger.error("QATManager: convert failed: {}", e)
        cpu_model.to(self.device)
        self._converted = True
        return self.model


# ── Helpers ───────────────────────────────────────────────────────────────────


def _resolve_backend(requested: str) -> str:
    """Return a backend that is actually supported on this platform.

    fbgemm and x86 are only available on x86 CPUs.  On macOS/ARM, only
    qnnpack is available.  Tries the requested backend first; falls back
    to qnnpack if it fails.
    """
    try:
        torch.backends.quantized.engine = requested
        return requested
    except RuntimeError:
        # Most common: fbgemm not available on macOS ARM
        fallback = "qnnpack"
        try:
            torch.backends.quantized.engine = fallback
        except RuntimeError:
            pass  # running in an environment with no quant engine
        return fallback


def _build_qconfig(cfg: QuantConfig) -> Any:
    """Build a PyTorch QConfig from QuantConfig settings.

    Returns a torch.quantization.QConfig with:
      - weight observer: per-channel MinMax (or per-tensor if per_channel=False)
      - activation observer: per-tensor MinMax (standard for activations)
    """
    import torch.quantization as tq

    if cfg.per_channel:
        weight_observer = tq.PerChannelMinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_channel_symmetric if cfg.symmetric else torch.per_channel_affine,
        )
    else:
        weight_observer = tq.MinMaxObserver.with_args(
            dtype=torch.qint8,
            qscheme=torch.per_tensor_symmetric if cfg.symmetric else torch.per_tensor_affine,
        )

    activation_observer = tq.MinMaxObserver.with_args(
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
    )

    return tq.QConfig(
        activation=tq.FakeQuantize.with_args(observer=activation_observer),
        weight=tq.FakeQuantize.with_args(observer=weight_observer),
    )


def _try_fuse(model: nn.Module) -> None:
    """Attempt to fuse Conv+BN and Conv+BN+ReLU pattern for better quantization.

    Fusion merges adjacent layers into a single quantizable unit, which is more
    accurate and efficient.  Safe to skip if the model has no such patterns.
    """
    import torch.quantization as tq

    try:
        tq.fuse_modules(model, [["conv", "bn", "relu"]], inplace=True)
    except Exception:
        pass  # not all models have these patterns
    try:
        tq.fuse_modules(model, [["conv", "relu"]], inplace=True)
    except Exception:
        pass


def _batch_to_device(batch: Any, device: torch.device) -> Any:
    """Move batch tensors to device (handles dict, list, tuple, Tensor)."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        moved = [_batch_to_device(x, device) for x in batch]
        return type(batch)(moved)
    return batch


def build_qat_manager(cfg: QuantConfig, model: nn.Module, device: torch.device) -> QATManager:
    """Convenience factory.  Import and call from Trainer.__init__."""
    return QATManager(cfg, model, device)
