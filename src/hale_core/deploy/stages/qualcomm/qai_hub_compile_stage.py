"""Compile (and optionally profile / validate) for Snapdragon devices on Qualcomm AI Hub.

AI Hub compiles in the cloud for a real device (``Samsung Galaxy S25``, ``Snapdragon X Elite
CRD`` ...) and returns a ready-to-ship asset:

=====================  ==========================================  =============================
``target_runtime``     asset                                        on-device runtime
=====================  ==========================================  =============================
``tflite``             ``.tflite`` (QNN delegate)                   LiteRT / TFLite (Android)
``qnn_dlc``            ``.dlc`` (hardware-agnostic QNN container)   QNN / Genie / QAIRT
``onnx``               ``.onnx`` (+ QNN EP options)                 ONNX Runtime QNN EP (Windows)
context binary         ``.bin`` via ``submit_compile_and_link_jobs``  QNN HTP (weight-shared)
=====================  ==========================================  =============================

(``qnn_context_binary`` / ``precompiled_qnn_onnx`` as compile targets were deprecated in 2026;
context binaries come from compile-and-link jobs.) Requires ``pip install qai-hub`` and an
API token (``qai-hub configure --api_token ...``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

import numpy as np
import torch
from loguru import logger

from hale_core.deploy.core.artifact import Artifact
from hale_core.deploy.core.export_context import ExportContext
from hale_core.deploy.core.optional_dependency import OptionalDependency
from hale_core.deploy.core.stage import TORCH_MODULE, Stage
from hale_core.deploy.exceptions import StageError
from hale_core.deploy.registry import STAGES

QAI_HUB = OptionalDependency(
    "qai_hub", "pip install qai-hub && qai-hub configure --api_token <token>"
)


@STAGES.register_module()
class QAIHubCompileStage(Stage):
    """Compile for ``device`` on AI Hub and download the asset; optionally profile and validate.

    Args:
        device: AI Hub device name (``hub.get_devices()`` lists them).
        chipset: alternatively select by chipset attribute (``qualcomm-snapdragon-8-elite``).
        source: ``torchscript`` (trace the export module) or ``onnx`` (latest ONNX artifact).
        target_runtime: ``tflite`` | ``qnn_dlc`` | ``onnx`` | ``context_binary``.
        quantize_full_type: ``int8`` | ``int16`` | ``w8a16`` | ``w4a8`` | ``w4a16`` (needs
            calibration data; not allowed for the ONNX target) or ``None`` for float.
        compute_unit: ``npu`` / ``gpu`` / ``cpu`` / ``all`` (comma-separated).
        qairt_version: ``major.minor``, ``default`` or ``latest``.
        extra_options: further compile options (string).
        profile: run a profile job and save its JSON (latency, memory, per-layer placement).
        validate_on_device: run an inference job and compare with the PyTorch reference.
        link_kwargs: extra kwargs of ``submit_compile_and_link_jobs`` (context binaries).
    """

    provides = ("qnn_model",)
    dependencies = (QAI_HUB,)

    def __init__(
        self,
        device: str | None = "Samsung Galaxy S25 (Family)",
        chipset: str | None = None,
        source: Literal["torchscript", "onnx"] = "torchscript",
        target_runtime: Literal["tflite", "qnn_dlc", "onnx", "context_binary"] = "qnn_dlc",
        quantize_full_type: str | None = None,
        compute_unit: str = "npu",
        qairt_version: str | None = None,
        extra_options: str = "",
        profile: bool = True,
        validate_on_device: bool = False,
        link_kwargs: Mapping[str, Any] | None = None,
        **base,
    ) -> None:
        super().__init__(**base)
        if device is None and chipset is None:
            raise ValueError("set device or chipset")
        self.device_name = device
        self.chipset = chipset
        self.source = source
        self.requires = ("onnx",) if source == "onnx" else (TORCH_MODULE,)
        self.target_runtime = target_runtime
        self.quantize_full_type = quantize_full_type
        self.compute_unit = compute_unit
        self.qairt_version = qairt_version
        self.extra_options = extra_options
        self.profile = profile
        self.validate_on_device = validate_on_device
        self.link_kwargs = dict(link_kwargs or {})

    # ------------------------------------------------------------------ options
    def compile_options(self, has_int64_io: bool) -> str:
        opts = (
            []
            if self.target_runtime == "context_binary"
            else [f"--target_runtime {self.target_runtime}"]
        )
        opts.append(f"--compute_unit {self.compute_unit}")
        if self.quantize_full_type:
            opts.append(f"--quantize_full_type {self.quantize_full_type}")
        if self.qairt_version:
            opts.append(f"--qairt_version {self.qairt_version}")
        if has_int64_io and self.target_runtime in ("tflite", "qnn_dlc", "context_binary"):
            opts.append("--truncate_64bit_io")  # NPU runtimes have no int64 tensors
        if self.extra_options:
            opts.append(self.extra_options)
        return " ".join(opts)

    def input_specs(self, ctx: ExportContext) -> dict[str, tuple[tuple[int, ...], str]]:
        sig = ctx.signature
        return {s.name: (sig.shape(s, "opt"), s.dtype) for s in sig.inputs}

    # ------------------------------------------------------------------ run
    def run(self, ctx: ExportContext) -> list[Artifact]:
        hub = QAI_HUB.load()
        out = ctx.stage_dir(self.name)
        device = (
            hub.Device(self.device_name)
            if self.device_name
            else hub.Device(attributes=f"chipset:{self.chipset}")
        )
        model = self._source_model(ctx, out)
        int64 = any(s.dtype == "int64" for s in ctx.signature.inputs)
        options = self.compile_options(int64)
        calibration = self._calibration(ctx) if self.quantize_full_type else None
        logger.info("AI Hub compile on {} with '{}'", self.device_name or self.chipset, options)
        if self.target_runtime == "context_binary":
            if not hasattr(hub, "submit_compile_and_link_jobs"):
                raise StageError(
                    self.name,
                    "this qai-hub client has no submit_compile_and_link_jobs; upgrade qai-hub",
                )
            jobs = hub.submit_compile_and_link_jobs(
                model,
                device=device,
                input_specs=self.input_specs(ctx),
                options=options,
                calibration_data=calibration,
                **self.link_kwargs,
            )
            job = jobs[-1] if isinstance(jobs, (list, tuple)) else jobs
        else:
            job = hub.submit_compile_job(
                model=model,
                device=device,
                input_specs=self.input_specs(ctx),
                options=options,
                calibration_data=calibration,
            )
        target_model = job.get_target_model()
        if target_model is None:
            raise StageError(
                self.name, f"AI Hub compile job {job.job_id} failed: {job.get_status()}"
            )
        asset = target_model.download(str(out / f"model.{self._suffix()}"))
        meta = {
            "job_id": job.job_id,
            "device": self.device_name or self.chipset,
            "options": options,
            "target_runtime": self.target_runtime,
            "quantize": self.quantize_full_type,
        }
        artifacts = [self.artifact(ctx, "qnn_model", asset, **meta)]
        if self.profile:
            profile = hub.submit_profile_job(model=target_model, device=device).download_profile()
            (out / "profile.json").write_text(json.dumps(profile, indent=2, default=str))
            artifacts.append(self.artifact(ctx, "qnn_profile", out / "profile.json", **meta))
        if self.validate_on_device:
            self._validate(ctx, hub, target_model, device, out)
        (out / "aihub.json").write_text(json.dumps(meta, indent=2))
        return artifacts

    def _source_model(self, ctx: ExportContext, out):
        if self.source == "onnx":
            return str(ctx.store.require("onnx").path)
        with torch.no_grad():
            traced = torch.jit.trace(ctx.export_module, ctx.example_args("opt"), check_trace=False)
        path = out / "model.pt"
        traced.save(str(path))
        return traced

    def _calibration(self, ctx: ExportContext) -> dict[str, list[np.ndarray]]:
        columns: dict[str, list[np.ndarray]] = {}
        for batch in ctx.calibration.as_numpy(ctx.signature):
            for name, array in batch.items():
                columns.setdefault(name, []).append(array)
        return columns

    def _validate(self, ctx: ExportContext, hub, target_model, device, out) -> None:
        from hale_core.deploy.validation.parity_checker import ParityChecker

        inputs = {k: [v.numpy()] for k, v in ctx.sample_inputs("opt").items()}
        job = hub.submit_inference_job(model=target_model, device=device, inputs=inputs)
        outputs = job.download_output_data()
        candidate = [np.asarray(v[0]) for v in outputs.values()]
        report = ParityChecker(atol=5e-2, min_cosine=0.99).check(
            list(ctx.signature.outputs),
            ctx.reference_outputs("opt"),
            candidate,
            f"aihub:{self.device_name}",
        )
        (out / "device_parity.json").write_text(json.dumps(report, indent=2))

    def _suffix(self) -> str:
        return {"tflite": "tflite", "qnn_dlc": "dlc", "onnx": "onnx.zip", "context_binary": "bin"}[
            self.target_runtime
        ]
