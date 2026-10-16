"""Deployment framework: export Hale models to NVIDIA, Apple, AMD and Qualcomm targets.

.. code-block:: text

    config.yaml (_base_) ─▶ DeployRunner ─▶ ModelAdapter ─▶ Pipeline[Stage, ...] ─▶ artifacts
                                                                     │
           NVIDIA  : ONNX → ModelOpt (AutoCast / PTQ) → TensorRT engine (10/11, RTX, Jetson DLA)
                     HF checkpoint → ModelOpt PTQ → TensorRT-LLM bundle;  Torch-TensorRT
           Apple   : torch.export → ATen→MLX bridge → .mlxfn + safetensors;  mlx-lm / mlx-vlm
           AMD     : ONNX → MIGraphX .mxr (Instinct / Radeon)
                     ONNX → Quark → Vitis AI EP (Ryzen AI NPU)
           Qualcomm: AI Hub compile (tflite / qnn_dlc / onnx / context binary) + profile
                     ExecuTorch QNN .pte

Quick start::

    from hale_core.deploy import DeployRunner
    report = DeployRunner.from_file("configs/deploy/gpt_tensorrt_fp16.yaml").run()

Public names are resolved lazily so ``import hale_core.deploy`` stays cheap and never imports
a vendor SDK.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "Config": "hale_core.deploy.config",
    "ConfigLoader": "hale_core.deploy.config",
    "DeployConfig": "hale_core.deploy.config",
    "DeployRunner": "hale_core.deploy.core.deploy_runner",
    "ExportContext": "hale_core.deploy.core.export_context",
    "IOSignature": "hale_core.deploy.core.io_signature",
    "Pipeline": "hale_core.deploy.core.pipeline",
    "Stage": "hale_core.deploy.core.stage",
    "Artifact": "hale_core.deploy.core.artifact",
    "ModelAdapter": "hale_core.deploy.adapters.model_adapter",
    "ExportModule": "hale_core.deploy.adapters.export_module",
    "Registry": "hale_core.deploy.registry",
    "STAGES": "hale_core.deploy.registry",
    "ADAPTERS": "hale_core.deploy.registry",
    "RUNTIMES": "hale_core.deploy.registry",
    "CALIBRATION": "hale_core.deploy.registry",
    "MLX_OPS": "hale_core.deploy.registry",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(_EXPORTS[name]), name)
