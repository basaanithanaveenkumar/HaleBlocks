"""The root registries of the deployment framework.

=================  =====================================================================
Registry           What is registered
=================  =====================================================================
``STAGES``         pipeline stages (``OnnxExportStage``, ``TensorRTEngineStage`` ...)
``ADAPTERS``       model adapters: how to build a model and its export signature
``CALIBRATION``    calibration data sources for post-training quantization
``RUNTIMES``       inference sessions used by parity checks (ORT, TensorRT, MLX ...)
``MLX_OPS``        ATen op → MLX converters used by the torch → MLX graph bridge
=================  =====================================================================

Downstream repos create child registries (``Registry("adapter", parent=ADAPTERS,
scope="hale_llm")``) or register directly into these.
"""

from __future__ import annotations

from hale_core.deploy.registry.registry import Registry

STAGES = Registry("stage")
ADAPTERS = Registry("adapter")
CALIBRATION = Registry("calibration")
RUNTIMES = Registry("runtime")
MLX_OPS = Registry("mlx_op")

__all__ = ["ADAPTERS", "CALIBRATION", "MLX_OPS", "RUNTIMES", "STAGES"]
