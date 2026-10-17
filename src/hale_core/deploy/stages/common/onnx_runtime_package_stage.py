"""Bundle an ONNX model with the ONNX Runtime execution-provider settings of the target device."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from typing import Any

from hale_core.deploy.core.artifact import Artifact
from hale_core.deploy.core.export_context import ExportContext
from hale_core.deploy.core.stage import Stage
from hale_core.deploy.registry import STAGES


@STAGES.register_module()
class OnnxRuntimePackageStage(Stage):
    """Copy the final ONNX (+ external data) and write ``deployment.json`` for the app side.

    ONNX Runtime is the common runtime of several targets in this project:

    =========================  ===============================  ===========================
    target                     execution provider               typical options
    =========================  ===============================  ===========================
    AMD Ryzen AI NPU           ``VitisAIExecutionProvider``     ``cache_dir``, ``cache_key``
    AMD Instinct / Radeon GPU  ``MIGraphXExecutionProvider``    ``device_id``, fp16 flags
    Snapdragon (Windows/Android) ``QNNExecutionProvider``       ``backend_path: QnnHtp``
    NVIDIA GPU                 ``TensorrtExecutionProvider``    ``trt_fp16_enable``
    =========================  ===============================  ===========================

    Args:
        providers: ordered EP names for the target.
        provider_options: per-provider options, aligned with ``providers``.
        session_options: free-form ORT session settings recorded for the app.
    """

    requires = ("onnx",)
    provides = ("onnx_package",)

    def __init__(
        self,
        providers: Sequence[str] = ("CPUExecutionProvider",),
        provider_options: Sequence[Mapping[str, Any]] | None = None,
        session_options: Mapping[str, Any] | None = None,
        **base,
    ) -> None:
        super().__init__(**base)
        self.providers = list(providers)
        self.provider_options = [dict(o) for o in (provider_options or [{} for _ in providers])]
        self.session_options = dict(session_options or {})

    def run(self, ctx: ExportContext) -> list[Artifact]:
        """Execute this stage on the pipeline context (see the class docstring).

        Args:
            ctx: the export context (model, signature, artifacts, work dir).

        Returns:
            The artifacts produced, also recorded in ``ctx.artifacts``.
        """
        source = ctx.store.require("onnx")
        out = ctx.stage_dir(self.name)
        target = out / "model.onnx"
        shutil.copy2(source.path, target)
        for data in source.path.parent.glob(f"{source.path.name}.data"):
            shutil.copy2(data, out / data.name)
        sig = ctx.signature
        deployment = {
            "model": target.name,
            "runtime": "onnxruntime",
            "providers": self.providers,
            "provider_options": self.provider_options,
            "session_options": self.session_options,
            "device": ctx.device,
            "inputs": [
                {"name": s.name, "shape": list(s.shape), "dtype": s.dtype} for s in sig.inputs
            ],
            "outputs": list(sig.outputs),
            "dims": {k: {"min": d.min, "opt": d.opt, "max": d.max} for k, d in sig.dims.items()},
        }
        (out / "deployment.json").write_text(json.dumps(deployment, indent=2))
        return [self.artifact(ctx, "onnx_package", out, providers=self.providers)]
