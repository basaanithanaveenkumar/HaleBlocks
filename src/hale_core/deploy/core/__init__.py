"""Pipeline core: signatures, artifacts, stages, the pipeline and the runner facade.

Public names resolve lazily (PEP 562) so importing one submodule (e.g. ``io_signature`` from an
adapter) never drags in the runner and its imports - this keeps the package free of import
cycles between ``core``, ``adapters`` and ``calibration``.
"""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "Artifact": "artifact",
    "ArtifactStore": "artifact_store",
    "CommandRunner": "command_runner",
    "DeployRunner": "deploy_runner",
    "DimSpec": "dim_spec",
    "ExportContext": "export_context",
    "IOSignature": "io_signature",
    "OptionalDependency": "optional_dependency",
    "Pipeline": "pipeline",
    "PipelineReport": "pipeline_report",
    "Stage": "stage",
    "TORCH_MODULE": "stage",
    "StageResult": "stage_result",
    "TensorSpec": "tensor_spec",
    "DTYPES": "tensor_spec",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)
