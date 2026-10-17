"""Report which deployment backends this machine can run (``hale-deploy env``)."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import os
import platform
import shutil

from hale_core.deploy.environment.backend_status import BackendStatus

#: (vendor, backend, import name or "bin:<executable>" / "env:<VAR>", distribution name, purpose)
PROBES: tuple[tuple[str, str, str, str | None, str], ...] = (
    ("common", "onnx", "onnx", "onnx", "graph interchange"),
    ("common", "onnxruntime", "onnxruntime", None, "ORT + execution providers"),
    ("common", "transformers", "transformers", "transformers", "HF checkpoints"),
    ("nvidia", "tensorrt", "tensorrt", None, "TensorRT engines"),
    ("nvidia", "tensorrt_rtx", "tensorrt_rtx", None, "TensorRT for RTX"),
    ("nvidia", "modelopt", "modelopt", "nvidia-modelopt", "AutoCast / PTQ"),
    ("nvidia", "torch_tensorrt", "torch_tensorrt", "torch-tensorrt", "Torch-TensorRT"),
    ("nvidia", "tensorrt_llm", "tensorrt_llm", "tensorrt-llm", "LLM serving"),
    ("apple", "mlx", "mlx", "mlx", "MLX arrays / export"),
    ("apple", "mlx_lm", "mlx_lm", "mlx-lm", "LLM conversion"),
    ("apple", "mlx_vlm", "mlx_vlm", "mlx-vlm", "VLM conversion"),
    ("amd", "migraphx", "migraphx", None, "MIGraphX (python)"),
    ("amd", "migraphx-driver", "bin:migraphx-driver", None, "MIGraphX (CLI)"),
    ("amd", "quark", "quark", "amd-quark", "Ryzen AI quantization"),
    ("qualcomm", "qai_hub", "qai_hub", "qai-hub", "AI Hub cloud compile"),
    ("qualcomm", "executorch", "executorch", "executorch", "ExecuTorch"),
    ("qualcomm", "QNN_SDK_ROOT", "env:QNN_SDK_ROOT", None, "QAIRT / QNN SDK"),
)


class EnvironmentChecker:
    """Probe every backend without importing heavy modules unless they exist."""

    def check(self) -> list[BackendStatus]:
        """Status of every known backend (import probes only; nothing is initialised)."""
        return [self._probe(*probe) for probe in PROBES]

    @staticmethod
    def _probe(
        vendor: str, backend: str, target: str, dist: str | None, purpose: str
    ) -> BackendStatus:
        if target.startswith("bin:"):
            path = shutil.which(target[4:])
            return BackendStatus(vendor, backend, path is not None, None, path or purpose)
        if target.startswith("env:"):
            value = os.environ.get(target[4:])
            return BackendStatus(vendor, backend, bool(value), None, value or purpose)
        if importlib.util.find_spec(target) is None:
            return BackendStatus(vendor, backend, False, None, purpose)
        version = None
        try:
            version = importlib.metadata.version(dist or target)
        except importlib.metadata.PackageNotFoundError:
            version = getattr(importlib.import_module(target), "__version__", None)
        return BackendStatus(vendor, backend, True, version, purpose)

    @staticmethod
    def accelerators() -> dict[str, str]:
        """Platform, torch version and detected accelerators (CUDA, MPS, ROCm …) as display strings."""
        import torch

        info = {
            "platform": f"{platform.system()} {platform.machine()}",
            "torch": torch.__version__,
            "cuda": str(torch.cuda.is_available()),
            "mps": str(torch.backends.mps.is_available()),
        }
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
        return info
