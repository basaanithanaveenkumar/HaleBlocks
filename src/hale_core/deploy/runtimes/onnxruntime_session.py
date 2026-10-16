"""ONNX Runtime session with a configurable execution-provider list."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from hale_core.deploy.core.optional_dependency import OptionalDependency
from hale_core.deploy.registry import RUNTIMES
from hale_core.deploy.runtimes.inference_session import InferenceSession

ORT = OptionalDependency(
    "onnxruntime", "pip install onnxruntime (or onnxruntime-gpu / onnxruntime-qnn)"
)


@RUNTIMES.register_module()
class OnnxRuntimeSession(InferenceSession):
    """Run an ONNX model with ORT.

    Args:
        path: ``.onnx`` file.
        providers: ordered EP names, e.g. ``["MIGraphXExecutionProvider", "CPUExecutionProvider"]``
            (AMD GPU), ``["VitisAIExecutionProvider", ...]`` (Ryzen AI NPU),
            ``["QNNExecutionProvider", ...]`` (Snapdragon), ``["TensorrtExecutionProvider", ...]``.
            Unavailable providers are dropped with a warning; CPU is always the last resort.
        provider_options: per-provider option dicts, aligned with ``providers``.
    """

    dependencies = (ORT,)

    def __init__(
        self,
        path: str | Path,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        provider_options: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        super().__init__(path)
        ort = ORT.load()
        available = set(ort.get_available_providers())
        options = list(provider_options or [{} for _ in providers])
        chosen = [(p, o) for p, o in zip(providers, options, strict=False) if p in available]
        dropped = [p for p in providers if p not in available]
        if dropped:
            logger.warning(
                "ORT providers {} unavailable here (have {})", dropped, sorted(available)
            )
        if not any(p == "CPUExecutionProvider" for p, _ in chosen):
            chosen.append(("CPUExecutionProvider", {}))
        self.session = ort.InferenceSession(
            str(self.path),
            providers=[p for p, _ in chosen],
            provider_options=[o for _, o in chosen],
        )
        self.input_names = [i.name for i in self.session.get_inputs()]

    def run(self, inputs: Mapping[str, np.ndarray]) -> list[np.ndarray]:
        return self.session.run(None, {name: inputs[name] for name in self.input_names})
